"""Does a fused time-loop kernel beat `lax.scan` on the LIF recurrence?

83% of an SHD training step is neuron loops, which `lax.scan` runs as T/unroll kernel launches
with every intermediate round-tripping through HBM. A Pallas kernel walks time inside a single
launch with the membrane resident in registers.

Forward only, one layer, no gradients -- the smallest experiment that answers the question.

Requires compute capability 8.0 or higher: the Triton backend refuses sm_75 and the Mosaic GPU
backend targets Hopper.

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
from jax.experimental.pallas import triton as plt

import jaxpike as jp


def _lif_kernel(x_ref, s_ref, *, timesteps: int, alpha: float, threshold: float, block: int):
    """One program owns a tile of neurons and walks the whole sequence for them.

    The membrane never leaves registers, and each step reads one contiguous row of the tile,
    so the loop costs one coalesced load and one store per timestep instead of a kernel launch.
    """
    start = pl.program_id(0) * block

    def step(t, v):
        window = (pl.ds(t, 1), pl.ds(start, block))
        x = x_ref[window].reshape(block)
        v = alpha * v + (1.0 - alpha) * x
        s = jnp.where(v > threshold, 1.0, 0.0)
        s_ref[window] = s.reshape(1, block)
        return v - threshold * s

    jax.lax.fori_loop(0, timesteps, step, jnp.zeros(block, dtype=x_ref.dtype))


@functools.partial(jax.jit, static_argnames=("alpha", "threshold", "block"))
def pallas_lif(x, *, alpha: float, threshold: float, block: int = 256):
    """`x` is (T, rows); returns spikes of the same shape.

    The tile stays in global memory and one row is loaded per timestep, which is all the
    recurrence needs live; a shaped `BlockSpec` would stage the entire sequence in SMEM.
    """
    timesteps, rows = x.shape
    if rows % block:
        raise ValueError(f"{rows} rows is not divisible by block {block}")
    # Triton rather than Mosaic GPU, which targets Hopper.
    spec = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        functools.partial(
            _lif_kernel, timesteps=timesteps, alpha=alpha, threshold=threshold, block=block
        ),
        grid=(rows // block,),
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        compiler_params=plt.CompilerParams(),
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
    ap.add_argument("--blocks", default="32,64,128,256")
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

    # A BlockSpec stages its whole tile in shared memory, bounding the block at
    # 64 KB / (T * 4 bytes * 2 buffers).
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
