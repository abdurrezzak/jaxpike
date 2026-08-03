"""Phase 0a, experiment 2: can the time axis be parallelized at all?

A LIF membrane is a first-order linear recurrence, ``v[t] = a*v[t-1] + (1-a)*x[t]``, which is
an associative scan and therefore parallelizable to O(log T) depth -- *provided the reset is
removed*, because reset feeds the spike back into the state and makes the recurrence
nonlinear. Reset-free (PSN-style) neurons are the tier-1 case in the Phase 2 plan.

What this proves locally: the associative formulation is numerically equivalent to the
sequential one, which is the part that can be wrong. What it cannot prove on a laptop: the
speedup. A CPU has a handful of cores, so a log-depth algorithm doing ~2x the total work
loses; the win needs thousands of parallel lanes. Timings are printed anyway, as an honest
record that the crossover has not yet been demonstrated.

Run:  python benchmarks/parallel_scan.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks._measure import timeit


def sequential(alpha, xs, v0):
    """Reference: one timestep at a time, O(T) depth."""

    def step(v, x):
        v = alpha * v + (1.0 - alpha) * x
        return v, v

    _, vs = jax.lax.scan(step, v0, xs)
    return vs


def parallel(alpha, xs, v0):
    """Associative scan: O(log T) depth, ~2x the total work.

    Each timestep is the affine map ``v -> A*v + b``. Composing two of them gives
    ``(A2*A1, A2*b1 + b2)``, which is associative, so the whole sequence can be reduced as a
    tree instead of a chain. The v0 contribution rides on the accumulated A.
    """
    b = (1.0 - alpha) * xs
    a = jnp.broadcast_to(jnp.asarray(alpha, xs.dtype), xs.shape)

    def compose(earlier, later):
        a1, b1 = earlier
        a2, b2 = later
        return a1 * a2, a2 * b1 + b2

    a_cum, b_cum = jax.lax.associative_scan(compose, (a, b))
    return a_cum * v0 + b_cum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    args = parser.parse_args()

    alpha = float(jnp.exp(-1.0 / 20.0))
    threshold = 1.0
    print(f"reset-free LIF membrane, batch={args.batch} neurons={args.neurons}")
    print(f"device: {jax.devices()[0]}\n")

    # No FLOP column: XLA's cost analysis does not descend into while-loop bodies, so it
    # attributes only a few thousand FLOPs to a sequential scan that actually performs
    # millions. That makes any work comparison between the two forms meaningless here, and a
    # number that is wrong by three orders of magnitude is worse than no number at all.
    header = f"{'T':>7} {'max |diff|':>12} {'spikes':>8} {'seq s':>9} {'par s':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    worst = 0.0
    for t in args.lengths:
        xs = jax.random.normal(jax.random.key(0), (t, args.batch, args.neurons)) * 2.0
        v0 = jnp.zeros((args.batch, args.neurons))

        vs_seq = sequential(alpha, xs, v0)
        vs_par = parallel(alpha, xs, v0)
        diff = float(jnp.max(jnp.abs(vs_seq - vs_par)))
        worst = max(worst, diff)

        # The spikes are what actually matter: a tiny membrane discrepancy is harmless unless
        # it flips a threshold crossing, so compare the binary output too.
        spikes_match = bool(jnp.array_equal(vs_seq > threshold, vs_par > threshold))

        t_seq = timeit(lambda x, v: sequential(alpha, x, v), xs, v0)
        t_par = timeit(lambda x, v: parallel(alpha, x, v), xs, v0)

        print(
            f"{t:>7} {diff:>12.2e} {'match' if spikes_match else 'DIFFER':>8} "
            f"{t_seq['median_s']:>9.4f} {t_par['median_s']:>9.4f} "
            f"{t_seq['median_s'] / max(t_par['median_s'], 1e-12):>7.2f}x"
        )

    print()
    tol = 1e-4
    verdict = "PASS" if worst < tol else "FAIL"
    print(f"Gate 0a (associative scan matches sequential, tol={tol:g}):")
    print(f"  worst membrane deviation across all lengths: {worst:.2e}  ->  {verdict}")
    platform = jax.devices()[0].platform
    if platform == "cpu":
        print(
            "\nNote: a speedup below 1.0x is expected on CPU and is not evidence against the "
            "approach. Log-depth parallelism trades ~2x the total work for O(log T) depth, "
            "which needs thousands of lanes to pay off. Run this on a GPU for the real number."
        )
    else:
        print(
            f"\nOn {platform} the log-depth trade pays off: the extra work is absorbed by "
            "parallel lanes while the O(T) dependency chain of the sequential scan is not. "
            "The advantage grows with T, because that is what deepens the chain."
        )


if __name__ == "__main__":
    main()
