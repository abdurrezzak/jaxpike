"""Running a network over time.

This module is the seam the whole performance plan hangs on. `unroll` is the sequential
reference: one `lax.scan` step per timestep, membrane state materialized for every step so
BPTT can reach it, costing O(T*B*N) memory. Phase 2 adds fused-kernel and parallel-in-time
backends behind this same signature, so user code does not change when the fast paths land.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def unroll(net, xs: Float[Array, "T *rest"], state=None) -> tuple[Array, object]:
    """Run `net` over the leading (time) axis of `xs`.

    Returns the per-timestep outputs stacked on the leading axis, plus the final state, so a
    long sequence can be processed in chunks by feeding the returned state back in.
    """
    if state is None:
        state = net.init_state(xs.shape[1:])

    def step(carry, x):
        return net(carry, x)

    final_state, ys = jax.lax.scan(step, state, xs)
    return ys, final_state


def spike_rate(spikes: Float[Array, "T *rest"]) -> Array:
    """Mean firing rate over time -- the standard rate-coded readout."""
    return jnp.mean(spikes, axis=0)


def spike_count(spikes: Float[Array, "T *rest"]) -> Array:
    return jnp.sum(spikes, axis=0)


def density(spikes: Array) -> Array:
    """Fraction of (timestep, neuron) slots that carried a spike.

    Reported by the benchmarks because it decides whether the Phase 2 sparse path can win:
    below roughly 10% density, gather-and-accumulate beats a dense matmul.
    """
    return jnp.mean(spikes != 0)


__all__ = ["density", "spike_count", "spike_rate", "unroll"]
