"""Parallel-in-time execution.

Sequential unrolling is O(T) deep: timestep t cannot start until t-1 finishes, so a GPU with
thousands of idle lanes waits on a dependency chain. When a neuron's recurrence is *linear*
the chain can be collapsed to O(log T) with an associative scan, which measured 119x faster
than sequential at T=8192 on a T4 (see benchmarks/README.md).

The requirement is linearity, and reset is what breaks it: a spike feeds back into the state,
so the recurrence stops being an affine map. Neurons that opt in publish their coefficients
via `recurrence()`; everything else raises rather than silently falling back, because a
silent fallback to O(T) would turn a 119x speedup into a performance mystery.

A feedforward network parallelizes end to end: `Dense` is applied independently at each
timestep, so it can process the whole time axis in one matmul, and the neuron layers use the
scan. Nothing in a feedforward stack forces sequential execution.
"""

from __future__ import annotations

import jax
from jaxtyping import Array, Float


def affine_compose(earlier, later):
    """Compose two affine maps ``v -> a*v + b``, oldest-first.

    ``(a1, b1)`` then ``(a2, b2)`` is ``(a1*a2, a2*b1 + b2)``. Associativity is what lets the
    sequence be reduced as a tree instead of walked as a chain.
    """
    a1, b1 = earlier
    a2, b2 = later
    return a1 * a2, a2 * b1 + b2


def scan_linear_recurrence(a, b, v0):
    """Solve ``v[t] = a[t]*v[t-1] + b[t]`` for all t at once, in O(log T) depth."""
    a_cum, b_cum = jax.lax.associative_scan(affine_compose, (a, b))
    return a_cum * v0 + b_cum


def supports_parallel(layer) -> bool:
    return hasattr(layer, "parallel_apply")


def unroll_parallel(net, xs: Float[Array, "T *rest"], state=None):
    """Run `net` over the whole time axis without a sequential dependency chain.

    Drop-in for `unroll` with the same signature and identical results, but every layer must
    support parallel execution. Raises `TypeError` naming the offending layer otherwise.
    """
    if state is None:
        state = net.init_state(xs.shape[1:])
    if not supports_parallel(net):
        raise TypeError(
            f"{type(net).__name__} has no parallel_apply, so it cannot run parallel-in-time. "
            "Reset-based neurons are nonlinear in time; use unroll() or "
            "unroll_checkpointed() for those, or a reset-free neuron such as LinearLIF."
        )
    # parallel_apply follows the (state, output) convention of the per-step __call__; unroll
    # returns (output, state). Swap here so the two entry points are interchangeable.
    final_state, ys = net.parallel_apply(state, xs)
    return ys, final_state


__all__ = [
    "affine_compose",
    "scan_linear_recurrence",
    "supports_parallel",
    "unroll_parallel",
]
