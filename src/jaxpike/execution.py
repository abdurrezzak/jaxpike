"""Running a network over time.

`unroll` is the sequential reference: one `lax.scan` step per timestep, membrane state
materialized for every step so BPTT can reach it, costing O(T*B*N) memory.
`unroll_checkpointed` trades recomputation for memory, and `jaxpike.parallel.unroll_parallel`
solves the time axis at once for reset-free models. All three share this signature.
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


def _best_chunk(t: int) -> int:
    """The divisor of `t` closest to sqrt(t), where two-level checkpointing is cheapest.

    Exact divisors only, rather than padding: padded timesteps would still advance the
    recurrence and silently corrupt the returned final state.
    """
    target = max(1, round(t**0.5))
    divisors = [d for d in range(1, t + 1) if t % d == 0]
    return min(divisors, key=lambda d: (abs(d - target), d))


def unroll_checkpointed(
    net, xs: Float[Array, "T *rest"], state=None, chunk_size: int | None = None
):
    """`unroll` with rematerialization: trades recomputation for memory.

    Identical results to `unroll`, but the backward pass re-runs each chunk's forward instead
    of keeping every timestep's residuals live, taking peak scratch memory from O(T*B*N) to
    roughly O(sqrt(T)*B*N) at the cost of one extra forward pass.
    """
    t = xs.shape[0]
    if state is None:
        state = net.init_state(xs.shape[1:])
    chunk = chunk_size if chunk_size is not None else _best_chunk(t)
    if t % chunk:
        raise ValueError(f"chunk_size {chunk} must divide the sequence length {t}")

    xs_chunked = xs.reshape(t // chunk, chunk, *xs.shape[1:])

    @jax.checkpoint
    def run_chunk(carry, chunk_xs):
        return jax.lax.scan(lambda c, x: net(c, x), carry, chunk_xs)

    final_state, ys = jax.lax.scan(run_chunk, state, xs_chunked)
    return ys.reshape(t, *ys.shape[2:]), final_state


def spike_rate(spikes: Float[Array, "T *rest"]) -> Array:
    """Mean firing rate over time -- the standard rate-coded readout."""
    return jnp.mean(spikes, axis=0)


def spike_count(spikes: Float[Array, "T *rest"]) -> Array:
    return jnp.sum(spikes, axis=0)


def density(spikes: Array) -> Array:
    """Fraction of (timestep, neuron) slots that carried a spike."""
    return jnp.mean(spikes != 0)


__all__ = ["density", "spike_count", "spike_rate", "unroll", "unroll_checkpointed"]
