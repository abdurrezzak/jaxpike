"""How many timesteps should the time loop emit per iteration?

A neuron step is a few elementwise ops on `(batch, N)`. At realistic sizes that is far too
little work to hide a kernel launch, so `lax.scan` spends its time dispatching rather than
computing. `unroll=k` emits k steps per iteration and lets XLA fuse them into one kernel.

The tradeoff is compile time and instruction cache pressure, both of which grow with k while
the speedup saturates. This sweeps k on the real SHD training step to fix the default.

    PYTHONPATH=. python benchmarks/scan_unroll.py --factors 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jaxpike as jp
from benchmarks.spyx_shd import build, integral_crossentropy


def timed_step(model, runner, factor: int, batch: int, timesteps: int, channels: int):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.adam(5e-4)
    opt_state = optimizer.init(params)

    def loss_fn(p, xs, labels):
        traces, _ = runner(eqx.combine(p, static), xs, scan_unroll=factor)
        return integral_crossentropy(traces, labels)

    @eqx.filter_jit
    def step(p, state, xs, labels):
        loss, grads = jax.value_and_grad(loss_fn)(p, xs, labels)
        updates, state = optimizer.update(grads, state, p)
        return eqx.apply_updates(p, updates), state, loss

    key = jax.random.key(0)
    xs = (jax.random.uniform(key, (timesteps, batch, channels)) < 0.05).astype(jnp.float32)
    labels = jax.random.randint(key, (batch,), 0, 20)

    start = time.perf_counter()
    params, opt_state, loss = step(params, opt_state, xs, labels)
    loss.block_until_ready()
    compile_seconds = time.perf_counter() - start

    times = []
    for _ in range(20):
        start = time.perf_counter()
        params, opt_state, loss = step(params, opt_state, xs, labels)
        loss.block_until_ready()
        times.append(time.perf_counter() - start)
    return compile_seconds, float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factors", default="1,2,4,8,16,32,64")
    ap.add_argument("--variant", default="sequential")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    args = ap.parse_args()

    runner = {"sequential": jp.unroll, "checkpointed": jp.unroll_checkpointed}[args.variant]
    print(f"device: {jax.devices()[0]}  variant {args.variant}")
    print(f"batch {args.batch}  T {args.timesteps}  hidden {args.hidden}")
    print(f"{'unroll':>7}  {'compile':>9}  {'step':>9}  {'speedup':>8}")

    baseline = None
    for factor in [int(f) for f in args.factors.split(",")]:
        model = build(args.hidden, args.channels, jax.random.key(0), neuron_kind="lif")
        compile_seconds, step_seconds = timed_step(
            model, runner, factor, args.batch, args.timesteps, args.channels
        )
        baseline = baseline or step_seconds
        print(
            f"{factor:>7}  {compile_seconds:>8.2f}s  {step_seconds * 1e3:>8.2f}ms  "
            f"{baseline / step_seconds:>7.2f}x"
        )


if __name__ == "__main__":
    main()
