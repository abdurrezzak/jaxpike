"""Phase 2 research: is parallel-in-time possible for reset-based (nonlinear) LIF?

Reset feeds a spike back into the membrane, so the recurrence stops being affine and the
associative scan that gives LinearLIF its 119x no longer applies. This measures a candidate:

    chunked fixed-point -- sequential over chunks with an exact carry, and within each chunk
    iterate "guess the spike train -> solve the resulting linear system in parallel ->
    recompute spikes" until the spikes stop changing.

The fixed point is the exact solution: if recomputing spikes from v reproduces the spikes
that produced v, the pair is consistent and equals what the sequential scan would give. That
is a verifiable certificate, not an approximation.

The open question is speed, since each chunk needs several parallel scans where the
sequential path needs one pass. Run:

    PYTHONPATH=. python benchmarks/reset_parallel.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from benchmarks._measure import timeit
from jaxpike import LIF, unroll
from jaxpike.parallel import scan_linear_recurrence


def _solve_given_spikes(alpha, b0, s, v0, threshold):
    """Membrane implied by an assumed spike train: the reset becomes a known input term."""
    s_prev = jnp.concatenate([jnp.zeros_like(s[:1]), s[:-1]], axis=0)
    a = jnp.broadcast_to(alpha, b0.shape)
    return scan_linear_recurrence(a, b0 - alpha * threshold * s_prev, v0)


def chunked_fixed_point(xs, v0, *, alpha, threshold, chunk, iters):
    """Exact reset-LIF: parallel within chunks, sequential across them.

    Both loops are `lax` constructs rather than Python loops. A Python loop here would be
    unrolled at trace time into `(T/chunk) * iters` separate scans -- at chunk=8, T=2048 that
    is 1280 of them, and compilation never finishes. `scan` and `fori_loop` keep the compiled
    graph the size of one chunk regardless of T.
    """
    t = xs.shape[0]
    if t % chunk:
        raise ValueError(f"chunk {chunk} must divide sequence length {t}")
    segments = xs.reshape(t // chunk, chunk, *xs.shape[1:])

    def run_chunk(carry, seg):
        b0 = (1.0 - alpha) * seg

        def refine(_, s):
            v = _solve_given_spikes(alpha, b0, s, carry, threshold)
            return (v > threshold).astype(seg.dtype)

        s = jax.lax.fori_loop(0, iters, refine, jnp.zeros_like(seg))
        v = _solve_given_spikes(alpha, b0, s, carry, threshold)
        return v[-1] - threshold * s[-1], s

    _, out = jax.lax.scan(run_chunk, v0, segments)
    return out.reshape(t, *out.shape[2:])


def iterations_to_converge(xs, v0, *, alpha, threshold, chunk, cap=128):
    """How many iterations each chunk needs before its spike train stops changing.

    Deliberately eager and host-synchronising: this is a measurement of the algorithm, not
    something that belongs on a hot path.
    """
    counts, carry = [], v0
    for start in range(0, xs.shape[0], chunk):
        seg = xs[start : start + chunk]
        b0 = (1.0 - alpha) * seg
        s = jnp.zeros_like(seg)
        used = cap
        for i in range(cap):
            v = _solve_given_spikes(alpha, b0, s, carry, threshold)
            s_new = (v > threshold).astype(seg.dtype)
            if jnp.array_equal(s_new, s):
                used = i + 1
                break
            s = s_new
        counts.append(used)
        carry = v[-1] - threshold * s[-1]
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--neurons", type=int, default=512)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--chunks", type=int, nargs="+", default=[8, 32, 128, 512])
    args = ap.parse_args()

    alpha = float(jnp.exp(-1.0 / 20.0))
    threshold = 1.0
    xs = (
        jnp.abs(jax.random.normal(jax.random.key(0), (args.length, args.batch, args.neurons))) * 3.0
    )
    v0 = jnp.zeros((args.batch, args.neurons))

    lif = LIF(tau=20.0, threshold=threshold, reset="subtract")
    ref, _ = unroll(lif, xs)
    seq = timeit(lambda x: unroll(lif, x)[0], xs)

    print(f"reset LIF, T={args.length} batch={args.batch} neurons={args.neurons}")
    print(f"device: {jax.devices()[0]}   spike density: {float(jnp.mean(ref)):.3f}")
    print(f"sequential scan: {seq['median_s']:.4f} s\n")

    hdr = f"{'chunk':>7} {'iters/chunk':>12} {'exact':>7} {'time s':>9} {'vs seq':>8}"
    print(hdr)
    print("-" * len(hdr))
    for chunk in args.chunks:
        counts = iterations_to_converge(xs, v0, alpha=alpha, threshold=threshold, chunk=chunk)
        need = max(counts)
        got = chunked_fixed_point(xs, v0, alpha=alpha, threshold=threshold, chunk=chunk, iters=need)
        exact = bool(jnp.array_equal(got, ref))
        mismatch = float(jnp.mean(got != ref))
        n_wrong = int(jnp.sum(got != ref))

        # Bind chunk/need as defaults: a bare closure would capture the loop variables and
        # every timing would silently use the last iteration's values.
        def run(x, chunk=chunk, need=need):
            return chunked_fixed_point(
                x, v0, alpha=alpha, threshold=threshold, chunk=chunk, iters=need
            )

        t = timeit(run, xs)
        ratio = seq["median_s"] / max(t["median_s"], 1e-12)
        print(
            f"{chunk:>7} {sum(counts) / len(counts):>7.1f} (max {need:>2}) "
            f"{'yes' if exact else 'NO':>6} {t['median_s']:>9.4f} {ratio:>7.2f}x"
            f"   wrong={n_wrong} ({mismatch:.2e})"
        )

    print(
        "\nThe fixed point is exact by construction -- consistency between the assumed and "
        "recomputed spike trains certifies it. Speed is the open question, and the tension is "
        "that larger chunks parallelize better but need more iterations."
    )


if __name__ == "__main__":
    main()
