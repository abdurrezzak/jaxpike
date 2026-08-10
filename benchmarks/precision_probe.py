"""Does lower-precision matrix multiplication close the gap to hand-written kernels?

Ablation puts 83% of an SHD training step in the neuron loops, but the remaining 31% is GEMMs,
and a T4 has fp16 tensor cores it never touches in an fp32 network. This measures the two
questions that decide whether mixed precision is worth building:

1. How much faster is the GEMM shape this network actually uses?
2. Does that survive into the end-to-end training step, where the neuron loop dominates?

Membrane state stays fp32 regardless. A leaky integrator runs for thousands of steps and
low-precision accumulation drifts enough to flip threshold crossings, which is a correctness
problem rather than a speed tradeoff.

    PYTHONPATH=. python benchmarks/precision_probe.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np


def timed(fn, *args, repeats: int = 50, warm_up: float = 2.0) -> float:
    jax.block_until_ready(fn(*args))
    deadline = time.perf_counter() + warm_up
    while time.perf_counter() < deadline:
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.min(times)) * 1e3


def gemm_sweep(rows: int, hidden: int, repeats: int) -> None:
    """The forward GEMM of one hidden layer, at the shape the SHD benchmark produces."""
    key = jax.random.key(0)
    x32 = jax.random.normal(key, (rows, hidden), jnp.float32)
    w32 = jax.random.normal(key, (hidden, hidden), jnp.float32)

    @jax.jit
    def fp32(x, w):
        return x @ w

    @jax.jit
    def fp16(x, w):
        return (x.astype(jnp.float16) @ w.astype(jnp.float16)).astype(jnp.float32)

    @jax.jit
    def fp16_resident(x, w):
        return x @ w

    baseline = timed(fp32, x32, w32, repeats=repeats)
    cast = timed(fp16, x32, w32, repeats=repeats)
    resident = timed(
        fp16_resident, x32.astype(jnp.float16), w32.astype(jnp.float16), repeats=repeats
    )

    error = float(jnp.max(jnp.abs(fp16(x32, w32) - fp32(x32, w32))))
    scale = float(jnp.max(jnp.abs(fp32(x32, w32))))

    print(f"  ({rows}, {hidden}) @ ({hidden}, {hidden})")
    print(f"    fp32                     {baseline:7.3f} ms")
    print(f"    fp16, casting each call  {cast:7.3f} ms   {baseline / cast:5.2f}x")
    print(f"    fp16, already resident   {resident:7.3f} ms   {baseline / resident:5.2f}x")
    print(f"    max abs error {error:.3e} against a peak magnitude of {scale:.1f}")


def step_sweep(batch: int, timesteps: int, hidden: int, channels: int, repeats: int) -> None:
    """The end-to-end training step, with the network's matmuls in each precision."""
    import equinox as eqx
    import optax

    import jaxpike as jp
    from benchmarks.spyx_shd import build, integral_crossentropy

    key = jax.random.key(0)
    xs = (jax.random.uniform(key, (timesteps, batch, channels)) < 0.05).astype(jnp.float32)
    labels = jax.random.randint(key, (batch,), 0, 20)
    optimizer = optax.adam(5e-4)

    def measure(dtype) -> float:
        model = build(hidden, channels, key, neuron_kind="lif")
        params, static = eqx.partition(model, eqx.is_inexact_array)
        opt_state = optimizer.init(params)

        def loss_fn(p, inputs, targets):
            combined = eqx.combine(p, static)
            if dtype is not jnp.float32:
                # Weights in reduced precision, membranes untouched: the neuron casts its
                # input back to STATE_DTYPE, so only the matmuls change.
                combined = jax.tree.map(
                    lambda leaf: leaf.astype(dtype) if leaf.ndim == 2 else leaf, combined
                )
                inputs = inputs.astype(dtype)
            traces, _ = jp.unroll(combined, inputs)
            return integral_crossentropy(traces, targets)

        @eqx.filter_jit
        def step(p, state, inputs, targets):
            loss, grads = jax.value_and_grad(loss_fn)(p, inputs, targets)
            updates, state = optimizer.update(grads, state, p)
            return eqx.apply_updates(p, updates), state, loss

        return timed(step, params, opt_state, xs, labels, repeats=repeats)

    baseline = measure(jnp.float32)
    reduced = measure(jnp.float16)
    print(f"  batch {batch}, T {timesteps}, hidden {hidden}")
    print(f"    fp32 weights   {baseline:7.3f} ms")
    print(f"    fp16 weights   {reduced:7.3f} ms   {baseline / reduced:5.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    print(f"device: {jax.devices()[0]}")
    print("\nisolated GEMM")
    for hidden in (128, 512):
        gemm_sweep(args.batch * args.timesteps, hidden, args.repeats)

    print("\nend-to-end training step")
    step_sweep(args.batch, args.timesteps, args.hidden, args.channels, args.repeats)


if __name__ == "__main__":
    main()
