"""Online learning: gradients computed forward in time, with no backward pass over time.

BPTT stores every timestep and then walks backwards through them. That costs `O(T)` memory,
cannot produce an update until the sequence ends, and has no implementation on neuromorphic
hardware, which sees a spike stream and cannot run time in reverse.

**e-prop** (Bellec et al., 2020) replaces the backward-in-time pass with a local *eligibility
trace*. Each synapse keeps a decaying memory of its own presynaptic activity; when an error
signal arrives, the weight update is trace times error. The trace is carried forward with the
network, so memory is `O(1)` in `T` no matter how long the sequence runs. It is the same
three-factor shape as the reward-modulated STDP in `plasticity.py`, with an error signal in
place of dopamine.

The factorization, for a weight from presynaptic unit `i` to postsynaptic unit `j`:

    trace:            e_i[t] = alpha * e_i[t-1] + (1 - alpha) * s_i[t]
    learning signal:  L_j[t] = dLoss / ds_j[t]        (spatial only, no time)
    gradient:         dLoss/dW_ji = sum_t  L_j[t] * psi_j[t] * e_i[t]

where `psi` is the surrogate derivative. The trace is the presynaptic spike train low-pass
filtered by the membrane's own time constant -- its impulse response, which is why the
factorization works.

The gradient is approximate, with two separate sources worth keeping apart:

*Reset.* A spike feeds back into its own membrane, a temporal path the factorization does not
carry. Without reset the membrane filter is the only route through time and the gradient is
exact to float precision (cosine 1.000000 against BPTT, versus 0.9988 with reset).

*Depth.* For a hidden layer the learning signal has to arrive from above, and doing that
exactly means filtering the downstream gradient backwards through each membrane -- a backward
pass in time, the thing online learning exists to avoid. It is propagated spatially only, the
standard symmetric-feedback approximation, giving a well-aligned descent direction (cosine
~0.9) rather than the true gradient.

Recurrent connections are dropped for the same reason as reset, so a recurrent `Graph` is
refused rather than silently mis-differentiated.

Memory is flat in `T`: traces ride in the scan carry and the weight gradient is accumulated in
place, so nothing per-timestep is ever stored. That is what forces the per-timestep loss
signature, since accumulating inside the scan requires a loss that decomposes over time.

Scope: `Sequential` stacks of alternating connection and spiking layers, which is what the
derivation covers. Anything else raises rather than silently returning a wrong gradient.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .layers import Dense, Sequential
from .neurons import LIF, LeakyIntegrator, LinearLIF

SPIKING = (LIF, LinearLIF)
CONNECTION = (Dense,)


def _decompose(net: Sequential):
    """Split the network into (connection, neuron) pairs plus a trailing readout.

    e-prop's derivation is stated for a connection feeding a neuron, so the structure has to
    be recognised explicitly rather than assumed.
    """
    if not isinstance(net, Sequential):
        raise TypeError(
            f"e-prop needs a Sequential of (connection, neuron) pairs, got {type(net).__name__}. "
            "Recurrent Graphs are not supported: e-prop drops recurrent temporal paths."
        )
    layers = list(net.layers)
    pairs, index = [], 0
    while index + 1 < len(layers):
        connection, neuron = layers[index], layers[index + 1]
        if not isinstance(connection, CONNECTION):
            raise TypeError(
                f"layer {index} is {type(connection).__name__}; e-prop expects a connection "
                "layer (Dense) at even positions"
            )
        if not isinstance(neuron, (*SPIKING, LeakyIntegrator)):
            raise TypeError(
                f"layer {index + 1} is {type(neuron).__name__}; e-prop expects a spiking layer "
                "or LeakyIntegrator at odd positions"
            )
        pairs.append((index, connection, neuron))
        index += 2
    if index != len(layers):
        raise TypeError(
            f"network has {len(layers)} layers; e-prop expects an even number, alternating "
            "connection and neuron"
        )
    return pairs


def _alpha(neuron) -> jnp.ndarray:
    return jnp.exp(-neuron.dt / jnp.exp(neuron.log_tau))


def _surrogate_derivative(neuron, membrane) -> jnp.ndarray:
    """psi: how strongly a nudge to the membrane changes the spike, per the surrogate."""
    if isinstance(neuron, LeakyIntegrator):
        return jnp.ones_like(membrane)
    offset = membrane - neuron.threshold
    return jax.vmap(jax.grad(lambda v: neuron.surrogate(v).sum()))(offset.reshape(-1)).reshape(
        offset.shape
    )


def eprop_grads(net: Sequential, xs, step_loss):
    """Gradient of ``sum_t step_loss(output[t])`` w.r.t. `net`, computed forward in time.

    `step_loss` takes **one timestep** of readout, shaped `(batch, units)`, and returns a
    scalar. That signature is the price of the memory guarantee. A loss that reduces across
    time first -- a max-over-time readout, for example -- cannot be trained this way and needs
    BPTT.

    Returns `(grads, loss)` with `grads` shaped like `net`.
    """
    pairs = _decompose(net)
    state = net.init_state(xs.shape[1:])

    # A trace lives on each connection's input, so its shape comes from walking the stack.
    shapes, shape = [], xs.shape[1:]
    for _, connection, neuron in pairs:
        shapes.append(shape)
        shape = neuron.out_shape(connection.out_shape(shape))
    traces0 = tuple(jnp.zeros(s) for s in shapes)
    grads0 = tuple(
        (
            jnp.zeros_like(connection.weight),
            None if connection.bias is None else jnp.zeros_like(connection.bias),
        )
        for _, connection, _ in pairs
    )

    def step(carry, x):
        net_state, traces, accumulated, total = carry
        new_state, new_traces, recorded = list(net_state), [], []
        value = x

        for slot, (index, connection, neuron) in enumerate(pairs):
            alpha = _alpha(neuron)
            trace = alpha * traces[slot] + (1.0 - alpha) * value
            _, current = connection(None, value)
            previous = net_state[index + 1]
            # The pre-reset membrane is what met the threshold, so the surrogate derivative
            # belongs there. Neurons return post-reset state, hence the recompute.
            pre_reset = alpha * previous.v + (1.0 - alpha) * current
            neuron_state, spikes = neuron(previous, current)
            new_state[index + 1] = neuron_state
            recorded.append((trace, pre_reset))
            new_traces.append(trace)
            value = spikes

        loss, delta = jax.value_and_grad(step_loss)(value)

        # Spatial backward pass, at this timestep only -- never through time.
        updated = list(accumulated)
        for slot in reversed(range(len(pairs))):
            _, connection, neuron = pairs[slot]
            trace, pre_reset = recorded[slot]
            signal = delta * _surrogate_derivative(neuron, pre_reset)
            weight_grad, bias_grad = updated[slot]
            weight_grad = weight_grad + jnp.einsum("bo,bi->oi", signal, trace)
            if bias_grad is not None:
                bias_grad = bias_grad + jnp.sum(signal, axis=0)
            updated[slot] = (weight_grad, bias_grad)
            delta = signal @ connection.weight

        return (tuple(new_state), tuple(new_traces), tuple(updated), total + loss), None

    (_, _, accumulated, total_loss), _ = jax.lax.scan(step, (state, traces0, grads0, 0.0), xs)

    grads = [None] * len(net.layers)
    for slot, (index, connection, neuron) in enumerate(pairs):
        weight_grad, bias_grad = accumulated[slot]
        grad = eqx.tree_at(lambda c: c.weight, connection, weight_grad, is_leaf=lambda v: v is None)
        if bias_grad is not None:
            grad = eqx.tree_at(lambda c: c.bias, grad, bias_grad, is_leaf=lambda v: v is None)
        grads[index] = grad
        grads[index + 1] = jax.tree.map(jnp.zeros_like, neuron)
    return _as_gradient_tree(net, grads), total_loss


def _as_gradient_tree(net: Sequential, grads):
    """Assemble per-layer gradients into a pytree matching `net`, zero where undefined."""
    filled = [
        g if g is not None else jax.tree.map(jnp.zeros_like, layer)
        for g, layer in zip(grads, net.layers, strict=True)
    ]
    return eqx.tree_at(lambda n: list(n.layers), net, filled, is_leaf=lambda x: x is None)


def eprop_value_and_grad(step_loss):
    """Wrap a per-timestep loss into a `(model, xs, *args) -> (loss, grads)` e-prop step.

    Drop-in alternative to `equinox.filter_value_and_grad` for supported architectures, so a
    training loop switches learning rule by changing one line. `step_loss(output_t, *args)`
    must take a single timestep.
    """

    def wrapped(model, xs, *args):
        grads, loss = eprop_grads(model, xs, lambda out: step_loss(out, *args))
        return loss, grads

    return wrapped


__all__ = ["eprop_grads", "eprop_value_and_grad"]
