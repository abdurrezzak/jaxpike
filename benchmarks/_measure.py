"""Measurement utilities shared by the benchmark scripts.

The memory measurement is the interesting one. Spyx's paper could not report memory because
JAX has no `torch.cuda.max_memory_allocated` equivalent, so the SNN benchmarking literature
compares speed and stays quiet about memory -- which is exactly where the biggest differences
are. XLA does know the answer, though: it reports the peak scratch allocation it planned for
a compiled executable. Reading that is device-independent, deterministic (no allocator noise,
no run-to-run variance), and works on a CPU-only laptop.

Caveat worth stating plainly: `temp_size_in_bytes` is XLA's *plan*, not an observed
high-water mark, and it excludes arguments and outputs. It is the right tool for comparing
how two implementations scale against each other, and the wrong tool for predicting whether
a specific model fits in a specific GPU. Phase 0b cross-checks it against real device
measurements.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import equinox as eqx
import jax


def as_array_fn(fn: Callable, module) -> tuple[Callable, object]:
    """Split an equinox module into (arrays, static) so plain `jax.jit` can be used.

    `eqx.filter_jit` returns a wrapper without `.lower().compile()`, so cost and memory
    analysis are unreachable through it.
    """
    params, static = eqx.partition(module, eqx.is_inexact_array)

    def wrapped(p, *args):
        return fn(eqx.combine(p, static), *args)

    return wrapped, params


def compile_stats(fn: Callable, *args) -> dict:
    """Peak scratch bytes and FLOPs for `fn(*args)` as XLA plans it."""
    compiled = jax.jit(fn).lower(*args).compile()
    mem = compiled.memory_analysis()
    try:
        cost = compiled.cost_analysis()
        flops = float(cost.get("flops", 0.0)) if isinstance(cost, dict) else 0.0
    except Exception:
        flops = 0.0
    return {
        "temp_bytes": int(mem.temp_size_in_bytes),
        "output_bytes": int(mem.output_size_in_bytes),
        "argument_bytes": int(mem.argument_size_in_bytes),
        "flops": flops,
    }


def timeit(fn: Callable, *args, repeats: int = 5, warmup: int = 2) -> dict:
    """Median wall-clock seconds, with compilation excluded and dispatch fully awaited."""
    compiled = jax.jit(fn)
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        samples.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "spread": (max(samples) - min(samples)) / max(min(samples), 1e-12),
    }


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1024
    return f"{n:,.1f} GB"
