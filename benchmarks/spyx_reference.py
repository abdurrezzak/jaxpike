"""Spyx's own SHD benchmark, run as a script so it can be timed beside jaxpike.

This is deliberately *their* code: the model, loss, optimizer and the fully-scanned training
loop are transcribed from `research/paper/SHD_jax.ipynb` in the Spyx repository, which is what
produced Table 1 of arXiv 2402.18994. Rewriting their training loop in a different style would
measure my transcription rather than their library.

Two changes, both to remove confounds rather than to help either side:

* Data comes in as arrays rather than through `tonic`, so both libraries see bit-identical
  inputs and neither pays a loader cost the other does not.
* The dataset is packed along time here instead of in a tonic transform, matching what their
  `_SHD2Raster` produced.

Requires `spyx==0.1.19` and `dm-haiku` -- the release contemporary with the paper. Later Spyx
moved to Flax NNX and the paper's API no longer exists.
"""

from __future__ import annotations

import argparse
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import spyx

N_CLASSES = 20


def build_snn(hidden: int, channels: int, sample_x):
    """Their `build_snn`: three biasless Linear layers, two LIF layers, an LI readout."""

    def shd_snn(x):
        core = hk.DeepRNN(
            [
                hk.Linear(hidden, with_bias=False),
                spyx.nn.LIF((hidden,), activation=spyx.axn.arctan()),
                hk.Linear(hidden, with_bias=False),
                spyx.nn.LIF((hidden,), activation=spyx.axn.arctan()),
                hk.Linear(N_CLASSES, with_bias=False),
                spyx.nn.LI((N_CLASSES,)),
            ]
        )
        # static unroll for maximum performance -- their comment, their setting
        spikes, V = hk.static_unroll(core, x, core.initial_state(x.shape[0]), time_major=False)
        return spikes, V

    key = jax.random.PRNGKey(0)
    SNN = hk.without_apply_rng(hk.transform(shd_snn))
    params = SNN.init(rng=key, x=jnp.float32(sample_x))
    return SNN, params


def train(SNN, params, x_train, y_train, *, epochs: int, batch_size: int, lr: float):
    """Their `benchmark`: epochs and batches are both `lax.scan`, all staged on device."""
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)
    Loss = spyx.fn.integral_crossentropy()

    @jax.jit
    def net_eval(weights, events, targets):
        traces, _final = SNN.apply(weights, events)
        return Loss(traces, targets)

    surrogate_grad = jax.value_and_grad(net_eval)
    rng = jax.random.PRNGKey(0)

    @jax.jit
    def train_step(state, data):
        grad_params, opt_state = state
        events, targets = data
        events = jnp.unpackbits(events, axis=1)  # decompress the temporal axis
        loss, grads = surrogate_grad(grad_params, events, targets)
        updates, opt_state = opt.update(grads, opt_state, grad_params)
        return [optax.apply_updates(grad_params, updates), opt_state], loss

    def shuffle(dataset, shuffle_rng, bs):
        x, y = dataset
        full = y.shape[0] // bs
        idx = jax.random.permutation(shuffle_rng, y.shape[0])[: full * bs]
        return (
            jnp.reshape(x[idx], (-1, bs, *x.shape[1:])),
            jnp.reshape(y[idx], (-1, bs)),
        )

    def epoch(epoch_state, epoch_num):
        curr_params, curr_opt_state = epoch_state
        shuffle_rng = jax.random.fold_in(rng, epoch_num)
        train_data = shuffle((x_train, y_train), shuffle_rng, batch_size)
        end_state, train_loss = jax.lax.scan(train_step, [curr_params, curr_opt_state], train_data)
        return end_state, jnp.mean(train_loss)

    final_state, metrics = jax.lax.scan(epoch, [params, opt_state], jnp.arange(epochs), epochs)
    return final_state[0], metrics


def accuracy(SNN, params, x, y, batch_size: int = 256) -> float:
    @jax.jit
    def batch_acc(weights, events, targets):
        events = jnp.unpackbits(events, axis=1)
        traces, _ = SNN.apply(weights, events)
        return jnp.sum(jnp.argmax(jnp.sum(traces, axis=1), axis=-1) == targets)

    correct = total = 0
    for start in range(0, len(y) - batch_size + 1, batch_size):
        correct += int(
            batch_acc(params, x[start : start + batch_size], y[start : start + batch_size])
        )
        total += batch_size
    return correct / max(total, 1)


def pack_time(dense: np.ndarray) -> np.ndarray:
    """(N, T, C) binary -> (N, T/8, C) bit-packed along time, as `_SHD2Raster` produced."""
    return np.packbits(dense, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--data", default="data/shd")
    args = ap.parse_args()

    # Reuse jaxpike's loader for the data so both libraries get identical inputs.
    from benchmarks.spyx_shd import load

    x_train, y_train = load("train", args.data, timesteps=args.timesteps, channels=args.channels)
    x_test, y_test = load("test", args.data, timesteps=args.timesteps, channels=args.channels)

    xt = jnp.asarray(pack_time(x_train))
    yt = jnp.asarray(y_train, dtype=jnp.uint8)
    xe = jnp.asarray(pack_time(x_test))
    ye = jnp.asarray(y_test, dtype=jnp.uint8)
    print(f"device: {jax.devices()[0]}")
    print(
        f"spyx {spyx.__version__ if hasattr(spyx, '__version__') else '?'}  packed train {xt.shape}"
    )

    sample = jnp.unpackbits(xt[: args.batch], axis=1)
    SNN, params = build_snn(args.hidden, args.channels, sample)

    trained, metrics = train(
        SNN, params, xt, yt, epochs=args.epochs, batch_size=args.batch, lr=args.lr
    )
    test_acc = accuracy(SNN, trained, xe, ye)
    print(f"final train loss {float(metrics[-1]):.4f}   TEST ACCURACY: {test_acc:.4f}")

    if args.trials:
        times = []
        for trial in range(args.trials + 1):
            start = time.perf_counter()
            _, out = train(
                SNN, params, xt, yt, epochs=args.epochs, batch_size=args.batch, lr=args.lr
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
