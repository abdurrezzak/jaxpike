"""Running a network over time.

`unroll` is the sequential reference: one `lax.scan` step per timestep, membrane state
materialized for every step so BPTT can reach it, costing O(T*B*N) memory.
`unroll_checkpointed` trades recomputation for memory, and `jaxpike.parallel.unroll_parallel`
solves the time axis at once for reset-free models. All three share this signature.

Both sequential paths segment the network before scanning. A layer that carries no state --
`Dense`, `Conv2d`, `Pool2d` -- gives the same answer whether it is applied once per timestep
or once to the whole stacked sequence, so it is hoisted out of the scan and evaluated as a
single call over `(T*batch)` rows. Only the neurons stay inside the time loop. This is pure
reassociation, bit-identical to the naive per-timestep walk, but it turns T small
latency-bound GEMMs per layer into one large one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

BATCHED, SCANNED = "batched", "scanned"

# Timesteps emitted per time-loop iteration. Chosen by measurement; see benchmarks/README.md.
SCAN_UNROLL = 1


def _plan(net, input_shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]] | None:
    """Group a `Sequential`'s layers into maximal batched and scanned runs.

    Returns `None` for anything that is not a plain layer chain -- `Graph`, a bare neuron, a
    user container -- which then takes the unsegmented path.
    """
    layers = getattr(net, "layers", None)
    if layers is None or not isinstance(layers, tuple):
        return None

    plan: list[tuple[str, list[int]]] = []
    shape = input_shape
    for index, layer in enumerate(layers):
        try:
            stateless = layer.init_state(shape) is None and hasattr(layer, "parallel_apply")
            shape = layer.out_shape(shape)
        except (AttributeError, TypeError, ValueError):
            return None
        kind = BATCHED if stateless else SCANNED
        if plan and plan[-1][0] == kind:
            plan[-1][1].append(index)
        else:
            plan.append((kind, [index]))
    return [(kind, tuple(indices)) for kind, indices in plan]


def _scan_segment(layers: tuple, carry: tuple, xs: Array, unroll_factor: int):
    """One `lax.scan` over time through a run of stateful layers."""

    def step(state, x):
        new = []
        for layer, s in zip(layers, state, strict=True):
            s, x = layer(s, x)
            new.append(s)
        return tuple(new), x

    return jax.lax.scan(step, carry, xs, unroll=unroll_factor)


def _run_segmented(net, xs: Array, state, plan, unroll_factor: int) -> tuple[object, Array]:
    """Apply the plan to `xs`. Returns `(state, outputs)`, the `lax.scan` carry convention."""
    new_state = list(state)
    for kind, indices in plan:
        if kind == BATCHED:
            for index in indices:
                _, xs = net.layers[index].parallel_apply(None, xs)
        else:
            layers = tuple(net.layers[i] for i in indices)
            carry, xs = _scan_segment(layers, tuple(state[i] for i in indices), xs, unroll_factor)
            for slot, index in enumerate(indices):
                new_state[index] = carry[slot]
    return tuple(new_state), xs


def _scan_unroll(t: int, requested: int | None) -> int:
    """How many timesteps to emit per loop iteration.

    A neuron step is a handful of elementwise ops on `(batch, N)`; at realistic sizes the
    kernel launch costs more than the arithmetic. Emitting several steps per iteration lets
    XLA fuse them into one kernel. The gain flattens once the body is large enough to keep the
    device busy, while compile time keeps growing, so this caps rather than scaling with T.
    """
    if requested is not None:
        return max(1, min(requested, t))
    return max(1, min(SCAN_UNROLL, t))


def unroll(
    net, xs: Float[Array, "T *rest"], state=None, *, scan_unroll: int | None = None
) -> tuple[Array, object]:
    """Run `net` over the leading (time) axis of `xs`.

    Returns the per-timestep outputs stacked on the leading axis, plus the final state, so a
    long sequence can be processed in chunks by feeding the returned state back in.

    `scan_unroll` overrides how many timesteps the time loop emits per iteration; it changes
    compile time and speed, never results.
    """
    if state is None:
        state = net.init_state(xs.shape[1:])

    factor = _scan_unroll(xs.shape[0], scan_unroll)
    plan = _plan(net, xs.shape[1:])
    if plan is not None:
        final_state, ys = _run_segmented(net, xs, state, plan, factor)
        return ys, final_state

    def step(carry, x):
        return net(carry, x)

    final_state, ys = jax.lax.scan(step, state, xs, unroll=factor)
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
    net,
    xs: Float[Array, "T *rest"],
    state=None,
    chunk_size: int | None = None,
    *,
    scan_unroll: int | None = None,
):
    """`unroll` with rematerialization: trades recomputation for memory.

    Identical results to `unroll`, but the backward pass re-runs each chunk's forward instead
    of keeping every timestep's residuals live, taking peak scratch memory from O(T*B*N) to
    roughly O(sqrt(T)*B*N) at the cost of one extra forward pass.

    Chunking happens above the segmentation, not inside it, so a chunk's hoisted GEMMs are
    rematerialized along with its neurons. Batching them over `chunk` timesteps instead of all
    `T` keeps most of the speed while leaving the memory bound intact.
    """
    t = xs.shape[0]
    if state is None:
        state = net.init_state(xs.shape[1:])
    chunk = chunk_size if chunk_size is not None else _best_chunk(t)
    if t % chunk:
        raise ValueError(f"chunk_size {chunk} must divide the sequence length {t}")

    plan = _plan(net, xs.shape[1:])
    xs_chunked = xs.reshape(t // chunk, chunk, *xs.shape[1:])
    factor = _scan_unroll(chunk, scan_unroll)

    if plan is not None:
        run_chunk = jax.checkpoint(lambda carry, cxs: _run_segmented(net, cxs, carry, plan, factor))
    else:

        @jax.checkpoint
        def run_chunk(carry, chunk_xs):
            return jax.lax.scan(lambda c, x: net(c, x), carry, chunk_xs, unroll=factor)

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
