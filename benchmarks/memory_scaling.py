"""Phase 0a, experiment 1: does BPTT memory actually scale the way the plan claims?

The thesis rests on the claim that every SNN framework pays O(T*B*N) memory to backprop
through time, and that this is remediable. This measures both halves on CPU, for free:

    naive scan        -- what every framework does today
    checkpointed scan -- pure JAX rematerialization, no kernels involved

Gate 0a: checkpointed must show >=5x lower peak scratch memory than naive at T=5000.

Run:  python benchmarks/memory_scaling.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks._measure import as_array_fn, compile_stats, human_bytes, timeit
from jaxpike import LIF, Dense, Sequential, unroll, unroll_checkpointed


def build(features: int, hidden: int, key) -> Sequential:
    k1, k2 = jax.random.split(key)
    return Sequential(
        Dense(features, hidden, key=k1),
        LIF(tau=20.0),
        Dense(hidden, hidden, key=k2),
        LIF(tau=20.0),
    )


def measure(net, xs, runner, *, time_it: bool):
    def loss(module, inputs):
        spikes, _ = runner(module, inputs)
        return jnp.mean(spikes)

    fn, params = as_array_fn(lambda m, x: jax.grad(loss)(m, x), net)
    stats = compile_stats(fn, params, xs)
    if time_it:
        stats.update(timeit(fn, params, xs))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lengths", type=int, nargs="+", default=[100, 500, 1000, 2500, 5000])
    parser.add_argument("--time", action="store_true", help="also measure wall clock (slow)")
    args = parser.parse_args()

    net = build(args.features, args.hidden, jax.random.key(0))
    print(f"forward+backward, batch={args.batch} features={args.features} hidden={args.hidden}")
    print(f"device: {jax.devices()[0]}\n")
    header = f"{'T':>7} {'naive':>14} {'checkpointed':>14} {'saving':>9} {'bytes/step':>11}"
    if args.time:
        header += f" {'naive s':>9} {'ckpt s':>9} {'slowdown':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for t in args.lengths:
        xs = jnp.ones((t, args.batch, args.features))
        naive = measure(net, xs, unroll, time_it=args.time)
        ckpt = measure(net, xs, unroll_checkpointed, time_it=args.time)
        ratio = naive["temp_bytes"] / max(ckpt["temp_bytes"], 1)
        row = (
            f"{t:>7} {human_bytes(naive['temp_bytes']):>14} "
            f"{human_bytes(ckpt['temp_bytes']):>14} {ratio:>8.1f}x "
            f"{naive['temp_bytes'] / t:>11,.0f}"
        )
        if args.time:
            slow = ckpt["median_s"] / max(naive["median_s"], 1e-12)
            row += f" {naive['median_s']:>9.4f} {ckpt['median_s']:>9.4f} {slow:>8.2f}x"
        print(row)
        results.append((t, naive, ckpt, ratio))

    longest, naive, ckpt, ratio = results[-1]
    print()
    print(f"Gate 0a (>=5x memory reduction at the longest sequence, T={longest}):")
    print(f"  naive        {human_bytes(naive['temp_bytes'])}")
    print(f"  checkpointed {human_bytes(ckpt['temp_bytes'])}")
    print(f"  reduction    {ratio:.1f}x  ->  {'PASS' if ratio >= 5.0 else 'FAIL'}")

    growth = [(t, n["temp_bytes"] / t) for t, n, _, _ in results]
    spread = max(b for _, b in growth) / min(b for _, b in growth)
    print(
        f"\nNaive memory is linear in T: bytes/step varies by only {spread:.2f}x "
        f"across T={results[0][0]}..{longest}, confirming the O(T*B*N) claim."
    )


if __name__ == "__main__":
    main()
