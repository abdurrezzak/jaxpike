"""Phase 2, experiment: end-to-end parallel-in-time speedup for a whole network.

The earlier parallel_scan.py isolated a single membrane. This measures what a user actually
gets: a full feedforward SNN (Dense -> LinearLIF -> Dense -> LinearLIF), forward and
forward+backward, sequential versus parallel-in-time.

Run:  PYTHONPATH=. python benchmarks/network_parallel.py
"""

from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp

from benchmarks._measure import as_array_fn, compile_stats, human_bytes, timeit
from jaxpike import Dense, LinearLIF, Sequential, unroll, unroll_parallel


def build(features, hidden, key, threshold=0.2):
    k1, k2 = jax.random.split(key)
    return Sequential(
        Dense(features, hidden, key=k1),
        LinearLIF(tau=20.0, threshold=threshold),
        Dense(hidden, hidden, key=k2),
        LinearLIF(tau=20.0, threshold=threshold),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--features", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    args = ap.parse_args()

    net = build(args.features, args.hidden, jax.random.key(0))
    print(f"batch={args.batch} features={args.features} hidden={args.hidden}")
    print(f"device: {jax.devices()[0]}\n")
    hdr = (
        f"{'T':>7} {'fwd seq':>9} {'fwd par':>9} {'fwd x':>7} "
        f"{'bwd seq':>9} {'bwd par':>9} {'bwd x':>7} {'par mem':>10} {'density':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for t in args.lengths:
        xs = jax.random.normal(jax.random.key(1), (t, args.batch, args.features)) * 5.0

        fwd_s, p = as_array_fn(lambda m, x: unroll(m, x)[0], net)
        fwd_p, _ = as_array_fn(lambda m, x: unroll_parallel(m, x)[0], net)

        def grad_of(runner):
            def loss(m, x):
                return jnp.mean(runner(m, x)[0])

            return as_array_fn(lambda m, x: eqx.filter_grad(loss)(m, x), net)[0]

        bwd_s, bwd_p = grad_of(unroll), grad_of(unroll_parallel)

        ts, tp = timeit(fwd_s, p, xs), timeit(fwd_p, p, xs)
        bs, bp = timeit(bwd_s, p, xs), timeit(bwd_p, p, xs)
        mem = compile_stats(bwd_p, p, xs)["temp_bytes"]
        density = float(jnp.mean(fwd_p(p, xs)))

        print(
            f"{t:>7} {ts['median_s']:>9.4f} {tp['median_s']:>9.4f} "
            f"{ts['median_s'] / max(tp['median_s'], 1e-12):>6.1f}x "
            f"{bs['median_s']:>9.4f} {bp['median_s']:>9.4f} "
            f"{bs['median_s'] / max(bp['median_s'], 1e-12):>6.1f}x "
            f"{human_bytes(mem):>10} {density:>8.3f}"
        )


if __name__ == "__main__":
    main()
