"""Can a fused time-loop kernel beat `lax.scan` on the LIF recurrence?

This is the question the library was founded on, asked as cheaply as possible. The ablation
puts 83% of an SHD training step in the neuron loops, and `lax.scan` runs them as T/unroll
separate kernel launches with every intermediate round-tripping through HBM. A Pallas kernel
loops over time inside one launch with the membrane held in registers.

Forward only, one neuron layer, no gradients: enough to tell whether the approach is worth
building out, and not one line more than that.

    PYTHONPATH=. python benchmarks/pallas_probe.py --batch 256 --neurons 128 --timesteps 256
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import jaxpike as jp


def _lif_kernel(x_ref, s_ref, *, timesteps: int, alpha: float, threshold: float):
    """One program owns a tile of neurons and walks the whole sequence for them.

    The membrane never leaves registers, and each step reads one contiguous row of the tile,
    so the loop costs one coalesced load and one store per timestep instead of a kernel launch.
    """

    def step(t, v):
        x = x_ref[t, :]
        v = alpha * v + (1.0 - alpha) * x
        s = jnp.where(v > threshold, 1.0, 0.0)
        s_ref[t, :] = s
        return v - threshold * s

    jax.lax.fori_loop(0, timesteps, step, jnp.zeros_like(x_ref[0, :]))


@functools.partial(jax.jit, static_argnames=("alpha", "threshold", "block"))
def pallas_lif(x, *, alpha: float, threshold: float, block: int = 256):
    """`x` is (T, rows); returns spikes of the same shape."""
    timesteps, rows = x.shape
    if rows % block:
        raise ValueError(f"{rows} rows is not divisible by block {block}")
    return pl.pallas_call(
        functools.partial(_lif_kernel, timesteps=timesteps, alpha=alpha, threshold=threshold),
        grid=(rows // block,),
        in_specs=[pl.BlockSpec((timesteps, block), lambda i: (0, i))],
        out_specs=pl.BlockSpec((timesteps, block), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x)


@functools.partial(jax.jit, static_argnames=("alpha", "threshold", "unroll"))
def scan_lif(x, *, alpha: float, threshold: float, unroll: int = 8):
    def step(v, xt):
        v = alpha * v + (1.0 - alpha) * xt
        s = jnp.where(v > threshold, 1.0, 0.0)
        return v - threshold * s, s

    _, spikes = jax.lax.scan(step, jnp.zeros_like(x[0]), x, unroll=unroll)
    return spikes


def timed(fn, x, repeats: int, warm_up: float = 2.0) -> float:
    jax.block_until_ready(fn(x))
    deadline = time.perf_counter() + warm_up
    while time.perf_counter() < deadline:
        jax.block_until_ready(fn(x))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(x))
        times.append(time.perf_counter() - start)
    return float(np.min(times)) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--neurons", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--blocks", default="8,16,32,64")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    alpha, threshold = float(jp.LIF(tau=20.0).alpha), 1.0
    rows = args.batch * args.neurons
    key = jax.random.key(0)
    x = (jax.random.uniform(key, (args.timesteps, rows)) < 0.05).astype(jnp.float32)

    print(f"device: {jax.devices()[0]}")
    print(f"T {args.timesteps}  rows {rows} (batch {args.batch} x neurons {args.neurons})")

    scan = functools.partial(scan_lif, alpha=alpha, threshold=threshold)
    reference = scan(x)
    scan_ms = timed(scan, x, args.repeats)
    print(f"lax.scan(unroll=8): {scan_ms:.3f} ms")

    # A BlockSpec stages its whole tile in shared memory, so the neuron block is bounded by
    # 64 KB / (T * 4 bytes * 2 buffers) -- 32 columns at T=256. Sweeping rather than assuming.
    for block in [int(b) for b in args.blocks.split(",")]:
        if rows % block:
            continue
        fused = functools.partial(pallas_lif, alpha=alpha, threshold=threshold, block=block)
        try:
            candidate = fused(x)
            mismatch = float(jnp.max(jnp.abs(candidate - reference)))
        except Exception as exc:
            print(f"  block {block:>4}: unavailable -- {type(exc).__name__}: {exc}")
            continue
        if mismatch > 0:
            print(f"  block {block:>4}: disagrees with lax.scan by {mismatch:.3e}, not timed")
            continue
        fused_ms = timed(fused, x, args.repeats)
        print(f"  block {block:>4}: {fused_ms:8.3f} ms   {scan_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    main()
