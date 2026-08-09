"""Head-to-head against the Spyx SHD training benchmark.

Reproduces the protocol of Heckel & Nowotny, "Spyx: A Library for Just-In-Time Compiled
Optimization of Spiking Neural Networks" (arXiv 2402.18994), Table 1, so that jaxpike can be
timed against it on the *same* GPU. Their published numbers are an RTX A6000 and are not
comparable to anything measured elsewhere; `benchmarks/gpu/run_spyx_comparison.py` runs both
libraries in one container.

Their protocol, taken from `research/paper/SHD_jax.ipynb` in the Spyx repository rather than
from the paper prose:

* SHD downsampled to **128 input channels** and **256 timesteps**, rasterized to binary.
* `Linear(128, H, bias=False) -> LIF -> Linear(H, H, bias=False) -> LIF ->
  Linear(H, 20, bias=False) -> LI`, with H = 128 or 512.
* Adam at 5e-4, integral cross-entropy with 0.3 label smoothing, 100 epochs, fp32.
* The whole dataset staged on the accelerator; epochs and batches are both `lax.scan`.
* One warm-up run, then five timed trials.

Their neuron is not jaxpike's `LIF`, so this file defines the Spyx recurrence directly
against the state contract. That keeps the model identical across libraries -- comparing
against a different neuron would measure the model, not the implementation.

    PYTHONPATH=. python benchmarks/spyx_shd.py --variant sequential --epochs 100
    PYTHONPATH=. python benchmarks/spyx_shd.py --smoke        # tiny, synthetic, CPU
"""

from __future__ import annotations

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

import jaxpike as jp
from benchmarks._measure import compile_stats
from benchmarks.shd_data import N_CLASSES, fake, load
from jaxpike.neurons import STATE_DTYPE
from jaxpike.parallel import scan_linear_recurrence

# --- the Spyx model, expressed against jaxpike's state contract ------------------------------


class SpyxArcTan(jp.Surrogate):
    """Spyx's `axn.arctan(k)`: derivative ``1 / (pi * (1 + (k*pi*v)^2))``.

    jaxpike's own `ATan` differs by a constant factor, which rescales the effective learning
    rate, so the relaxation is written out here to keep the gradient bit-comparable.

    `k` is static so the surrogate contributes no leaves: the model rides in a `lax.scan`
    carry, and a Python float leaf would be promoted to a tracer on the first iteration and
    break the carry's structure.
    """

    k: float = eqx.field(static=True, default=2.0)

    def relaxation(self, v):
        k_pi = self.k * jnp.pi
        return jnp.arctan(k_pi * v) / (k_pi * jnp.pi) + 0.5


def _init_beta(key, shape) -> Array:
    """Spyx initializes decay as truncated_normal(stddev=0.25) + 0.5, clipped to [0, 1]."""
    beta = jax.random.truncated_normal(key, -2.0, 2.0, shape) * 0.25 + 0.5
    return jnp.asarray(beta, dtype=STATE_DTYPE)


class SpyxLIF(eqx.Module):
    """Spyx's LIF, exactly as implemented in `spyx.nn.LIF`.

        s[t] = H(v[t-1] - threshold)
        v[t] = beta*v[t-1] + x[t] - threshold*s[t]

    Two differences from jaxpike's `LIF` matter. The spike is read from the membrane
    *before* this step's input, so it lags by one timestep; and the input carries no
    ``(1 - beta)`` normalization, so weights are not in threshold units. `beta` is a learnable
    per-neuron parameter clipped to [0, 1] on use.
    """

    beta: Float[Array, " hidden"]
    surrogate: jp.Surrogate
    threshold: float = eqx.field(static=True)

    def __init__(self, hidden: int, *, key, threshold: float = 1.0, k: float = 2.0):
        self.beta = _init_beta(key, (hidden,))
        self.surrogate = SpyxArcTan(k=k)
        self.threshold = threshold

    def init_state(self, input_shape: tuple[int, ...]):
        return jnp.zeros(input_shape, dtype=STATE_DTYPE)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state, x):
        beta = jnp.clip(self.beta, 0.0, 1.0)
        spikes = self.surrogate(state - self.threshold)
        v = beta * state + x.astype(STATE_DTYPE) - self.threshold * spikes
        return v, spikes


class SpyxLinearLIF(eqx.Module):
    """`SpyxLIF` with the reset term dropped and nothing else changed.

    This isolates the reset: the spike still lags the membrane by a step, exactly as in
    `SpyxLIF`. That makes it the right model for answering "what does reset alone contribute",
    and the *wrong* model for "how good is a reset-free network", because no one builds a
    reset-free neuron that also throws away the current step's input. Use `SpyxPSU` for that.
    """

    beta: Float[Array, " hidden"]
    surrogate: jp.Surrogate
    threshold: float = eqx.field(static=True)

    def __init__(self, hidden: int, *, key, threshold: float = 1.0, k: float = 2.0):
        self.beta = _init_beta(key, (hidden,))
        self.surrogate = SpyxArcTan(k=k)
        self.threshold = threshold

    def init_state(self, input_shape: tuple[int, ...]):
        return jnp.zeros(input_shape, dtype=STATE_DTYPE)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state, x):
        beta = jnp.clip(self.beta, 0.0, 1.0)
        spikes = self.surrogate(state - self.threshold)
        return beta * state + x.astype(STATE_DTYPE), spikes

    def parallel_apply(self, state, xs):
        beta = jnp.clip(self.beta, 0.0, 1.0)
        a = jnp.broadcast_to(beta, xs.shape)
        v = scan_linear_recurrence(a, xs.astype(STATE_DTYPE), state)
        # The spike lags the membrane by one step, so shift the trace and prepend the carry.
        previous = jnp.concatenate([state[None], v[:-1]], axis=0)
        return v[-1], self.surrogate(previous - self.threshold)


class SpyxPSU(eqx.Module):
    """Reset-free LIF that integrates *before* spiking, matching Spyx's own `PSU_LIF`.

        v[t] = beta*v[t-1] + x[t]
        s[t] = H(v[t] - threshold)

    Spyx 1.0.0 ships this as `PSU_LIF` / `AssociativeLIF` (marked experimental) with a
    `jax.lax.associative_scan` path, so the reset-free parallel idea is no longer unique to
    jaxpike -- it was absent from 0.1.19, the release the paper benchmarked.

    Against `SpyxLinearLIF` this drops the one-step spike lag, which is a second difference
    from `SpyxLIF` and therefore not a clean reset ablation. It is, however, the model anyone
    would actually train, so it is the one a speed claim should be attached to.
    """

    beta: Float[Array, " hidden"]
    surrogate: jp.Surrogate
    threshold: float = eqx.field(static=True)

    def __init__(self, hidden: int, *, key, threshold: float = 1.0, k: float = 2.0):
        self.beta = _init_beta(key, (hidden,))
        self.surrogate = SpyxArcTan(k=k)
        self.threshold = threshold

    def init_state(self, input_shape: tuple[int, ...]):
        return jnp.zeros(input_shape, dtype=STATE_DTYPE)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state, x):
        v = jnp.clip(self.beta, 0.0, 1.0) * state + x.astype(STATE_DTYPE)
        return v, self.surrogate(v - self.threshold)

    def parallel_apply(self, state, xs):
        beta = jnp.clip(self.beta, 0.0, 1.0)
        a = jnp.broadcast_to(beta, xs.shape)
        v = scan_linear_recurrence(a, xs.astype(STATE_DTYPE), state)
        return v[-1], self.surrogate(v - self.threshold)


class SpyxLI(eqx.Module):
    """Spyx's leaky-integrator readout: ``v[t] = beta*v[t-1] + x[t]``, output is `v[t]`."""

    beta: Float[Array, " units"]

    def __init__(self, units: int, *, key):
        self.beta = _init_beta(key, (units,))

    def init_state(self, input_shape: tuple[int, ...]):
        return jnp.zeros(input_shape, dtype=STATE_DTYPE)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state, x):
        v = jnp.clip(self.beta, 0.0, 1.0) * state + x.astype(STATE_DTYPE)
        return v, v

    def parallel_apply(self, state, xs):
        beta = jnp.clip(self.beta, 0.0, 1.0)
        a = jnp.broadcast_to(beta, xs.shape)
        v = scan_linear_recurrence(a, xs.astype(STATE_DTYPE), state)
        return v[-1], v


NEURONS = {"lif": SpyxLIF, "linear": SpyxLinearLIF, "psu": SpyxPSU}


def build(hidden: int, in_channels: int, key, *, neuron_kind: str) -> jp.Sequential:
    """The Spyx benchmark network. Linear layers carry no bias, matching their notebook."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    neuron = NEURONS[neuron_kind]
    return jp.Sequential(
        jp.Dense(in_channels, hidden, key=k1, use_bias=False),
        neuron(hidden, key=k2),
        jp.Dense(hidden, hidden, key=k3, use_bias=False),
        neuron(hidden, key=k4),
        jp.Dense(hidden, N_CLASSES, key=k5, use_bias=False),
        SpyxLI(N_CLASSES, key=k6),
    )


def integral_crossentropy(traces, labels, smoothing: float = 0.3):
    """Spyx's loss: cross-entropy on the membrane integrated over time, with label smoothing."""
    logits = jnp.sum(traces, axis=0)  # time is the leading axis in jaxpike
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    smoothed = optax.smooth_labels(one_hot, smoothing)
    return jnp.mean(optax.softmax_cross_entropy(logits, smoothed))


# --- training, staged on the accelerator like theirs -------------------------------------------


def make_step(static, runner, optimizer):
    """One training step over a single batch.

    Only the array half of the model travels in the scan carry; `static` is closed over. A
    carry holding non-array leaves changes structure between iterations.
    """

    def loss_fn(params, xs, labels):
        traces, _ = runner(eqx.combine(params, static), xs)
        return integral_crossentropy(traces, labels)

    def step(carry, batch):
        params, opt_state = carry
        xs, labels = batch
        # (batch, time, channels) uint8 -> (time, batch, channels) float32. Done here rather
        # than per epoch so the cast is a fused prologue to the first GEMM instead of a
        # materialized buffer four times the size of the epoch's inputs.
        xs = jnp.swapaxes(xs, 0, 1).astype(jnp.float32)
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (eqx.apply_updates(params, updates), opt_state), loss

    return step


def train(model, runner, inputs, labels, *, epochs: int, batch_size: int, lr: float, key):
    """Train fully on device. `inputs` is (N, T, C) uint8; batches are time-major per step.

    Epochs and batches are both `lax.scan`, so the whole run is one dispatch -- the staging
    that Spyx's speed rests on, matched here so the comparison is of implementations rather
    than of training-loop overhead.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    step = make_step(static, runner, optimizer)

    n_batches = len(labels) // batch_size
    usable = n_batches * batch_size

    def run_epoch(carry, epoch_key):
        current, opt_state = carry
        order = jax.random.permutation(epoch_key, len(labels))[:usable]
        xs = inputs[order].reshape(n_batches, batch_size, *inputs.shape[1:])
        ys = labels[order].reshape(n_batches, batch_size)
        (current, opt_state), losses = jax.lax.scan(step, (current, opt_state), (xs, ys))
        return (current, opt_state), jnp.mean(losses)

    @jax.jit
    def run_all(params, opt_state, keys):
        return jax.lax.scan(run_epoch, (params, opt_state), keys)

    keys = jax.random.split(key, epochs)
    (params, _), losses = run_all(params, opt_state, keys)
    return eqx.combine(params, static), losses


def accuracy(model, runner, inputs, labels, batch_size: int = 256) -> float:
    @eqx.filter_jit
    def batch_acc(m, xs, ys):
        traces, _ = runner(m, xs)
        return jnp.sum(jnp.argmax(jnp.sum(traces, axis=0), axis=-1) == ys)

    correct, total = 0, 0
    for start in range(0, len(labels) - batch_size + 1, batch_size):
        xs = jnp.asarray(inputs[start : start + batch_size], dtype=jnp.float32)
        xs = jnp.swapaxes(xs, 0, 1)
        ys = jnp.asarray(labels[start : start + batch_size])
        correct += int(batch_acc(model, xs, ys))
        total += batch_size
    return correct / max(total, 1)


def layer_rates(model, runner, inputs, batch_size: int = 256) -> list[float]:
    """Firing rate after each spiking layer.

    Reset-free neurons cannot depress themselves after firing, so the failure mode to look for
    is saturation -- a layer firing on nearly every step carries no temporal information and
    sits where the surrogate gradient is flat.
    """
    xs = jnp.swapaxes(jnp.asarray(inputs[:batch_size], dtype=jnp.float32), 0, 1)
    rates = []
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tuple(NEURONS.values())):
            prefix = jp.Sequential(*model.layers[: index + 1])
            spikes, _ = runner(prefix, xs)
            rates.append(float(jp.density(spikes)))
    return rates


def peak_scratch_bytes(model, runner, batch_size: int, timesteps: int, channels: int) -> int:
    """XLA's planned peak scratch for one training step, the measure used elsewhere here."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step = make_step(static, runner, optimizer)
    xs = jnp.zeros((batch_size, timesteps, channels), jnp.uint8)
    ys = jnp.zeros((batch_size,), jnp.int32)

    def one(p, state, batch_xs, batch_ys):
        return step((p, state), (batch_xs, batch_ys))

    return compile_stats(one, params, opt_state, xs, ys)["temp_bytes"]


# Each variant names a runner and the neuron it is valid for. `linear` isolates reset;
# `parallel` uses the reset-free neuron anyone would actually train.
RUNNERS = {
    "sequential": (jp.unroll, "lif"),
    "checkpointed": (jp.unroll_checkpointed, "lif"),
    "linear-ablation": (jp.unroll_parallel, "linear"),
    "parallel": (jp.unroll_parallel, "psu"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(RUNNERS), default="sequential")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--trials", type=int, default=0, help="timed trials after one warm-up")
    ap.add_argument("--data", default="data/shd")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, tiny, for CPU")
    ap.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="time only; skips the extra training run used to report accuracy",
    )
    args = ap.parse_args()

    runner, neuron_kind = RUNNERS[args.variant]
    print(f"device: {jax.devices()[0]}")

    if args.smoke:
        x_train, y_train = fake(512, args.timesteps, args.channels, seed=0)
        x_test, y_test = fake(256, args.timesteps, args.channels, seed=1)
    else:
        x_train, y_train = load(
            "train", args.data, timesteps=args.timesteps, channels=args.channels
        )
        x_test, y_test = load("test", args.data, timesteps=args.timesteps, channels=args.channels)
    print(
        f"train {x_train.shape} {x_train.dtype}  test {x_test.shape}  density {x_train.mean():.4f}"
    )

    # Staged on the accelerator as uint8, matching Spyx; cast per batch inside the epoch.
    train_inputs = jnp.asarray(x_train)
    train_labels = jnp.asarray(y_train, dtype=jnp.int32)

    key = jax.random.key(0)
    model = build(args.hidden, args.channels, key, neuron_kind=neuron_kind)

    scratch = peak_scratch_bytes(model, runner, args.batch, args.timesteps, args.channels)
    print(
        f"variant {args.variant} ({neuron_kind})  hidden {args.hidden}  "
        f"batch {args.batch}  T {args.timesteps}"
    )
    print(f"peak scratch: {scratch / 2**20:.1f} MB")

    if not args.skip_accuracy:
        trained, losses = train(
            model,
            runner,
            train_inputs,
            train_labels,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            key=key,
        )
        test_acc = accuracy(trained, runner, x_test, y_test)
        rates = layer_rates(trained, runner, x_test)
        print(f"final train loss {float(losses[-1]):.4f}   TEST ACCURACY: {test_acc:.4f}")
        print("layer firing rates: " + ", ".join(f"{r:.3f}" for r in rates))

    if args.trials:
        times = []
        for trial in range(args.trials + 1):
            start = time.perf_counter()
            fresh = build(args.hidden, args.channels, key, neuron_kind=neuron_kind)
            _, out = train(
                fresh,
                runner,
                train_inputs,
                train_labels,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                key=key,
            )
            out.block_until_ready()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(
                f"  trial {trial}{' (warm-up, discarded)' if trial == 0 else ''}: {elapsed:.2f} s"
            )
        timed = np.array(times[1:])
        print(f"TIME {args.epochs} epochs: {timed.mean():.2f} +/- {timed.std():.2f} s")


if __name__ == "__main__":
    main()
