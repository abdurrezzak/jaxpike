"""e-prop tests.

The claims worth pinning are about *how close* the online gradient is to BPTT, and where the
approximation comes from. Two independent sources: reset (a spike feeding back into its own
membrane) and depth (the learning signal for a hidden layer arriving without its temporal
filtering). Cosine similarity is the meaningful metric — a learning rule needs a well-aligned
descent direction, not the exact gradient.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxpike import (
    ALIF,
    LIF,
    Dense,
    Graph,
    LeakyIntegrator,
    LinearLIF,
    Sequential,
    eprop_grads,
    eprop_value_and_grad,
    lif_gain,
    unroll,
)

GAIN = lif_gain(20.0)


def squared(output_t):
    """Per-timestep loss. e-prop needs the loss to decompose over time."""
    return jnp.mean(output_t**2)


def total_squared(outputs):
    """The same loss summed over time, for the BPTT reference."""
    return jnp.sum(jax.vmap(squared)(outputs))


def drive(t=40, batch=3, features=12, seed=7, scale=2.0):
    return jax.random.normal(jax.random.key(seed), (t, batch, features)) * scale


def bptt_grads(net, xs):
    return eqx.filter_grad(lambda m, x: total_squared(unroll(m, x)[0]))(net, xs)


def cosine(a, b):
    return float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def single_pair(neuron, features=12, units=6):
    return Sequential(Dense(features, units, key=jax.random.key(0), gain=GAIN), neuron)


def two_pairs(neuron, features=12, hidden=24, units=6):
    k = jax.random.split(jax.random.key(0), 2)
    return Sequential(
        Dense(features, hidden, key=k[0], gain=GAIN),
        neuron,
        Dense(hidden, units, key=k[1], gain=GAIN),
        LeakyIntegrator(tau=20.0),
    )


# --- exactness -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "neuron",
    [LinearLIF(tau=20.0, threshold=0.4), LeakyIntegrator(tau=20.0)],
    ids=["linear_lif", "leaky_integrator"],
)
def test_exact_for_reset_free_output_layer(neuron):
    """No reset and an exact learning signal means the membrane filter is the only temporal
    path, and e-prop recovers BPTT to float precision."""
    net, xs = single_pair(neuron), drive()
    exact = bptt_grads(net, xs).layers[0].weight
    online, _ = eprop_grads(net, xs, squared)
    got = online.layers[0].weight
    assert cosine(exact, got) == pytest.approx(1.0, abs=1e-5)
    assert jnp.allclose(exact, got, atol=1e-5, rtol=1e-3)


def test_reset_costs_only_a_little_accuracy_at_the_output_layer():
    net, xs = single_pair(LIF(tau=20.0, threshold=0.4)), drive()
    exact = bptt_grads(net, xs).layers[0].weight
    online, _ = eprop_grads(net, xs, squared)
    assert cosine(exact, online.layers[0].weight) > 0.99


def test_hidden_layers_stay_well_aligned_even_though_they_are_approximate():
    """Symmetric feedback: a usable descent direction, not the true gradient."""
    for neuron in (LinearLIF(tau=20.0, threshold=0.4), LIF(tau=20.0, threshold=0.4)):
        net, xs = two_pairs(neuron), drive()
        exact = bptt_grads(net, xs)
        online, _ = eprop_grads(net, xs, squared)
        assert cosine(exact.layers[2].weight, online.layers[2].weight) > 0.999
        hidden = cosine(exact.layers[0].weight, online.layers[0].weight)
        assert 0.7 < hidden < 1.0, f"hidden alignment {hidden} outside the expected band"


def test_gradient_is_a_descent_direction():
    """The property that actually matters: stepping along it decreases the loss."""
    net, xs = two_pairs(LinearLIF(tau=20.0, threshold=0.4)), drive()
    online, loss = eprop_grads(net, xs, squared)
    stepped = jax.tree.map(lambda p, g: p - 1e-3 * g if eqx.is_inexact_array(p) else p, net, online)
    assert squared(unroll(stepped, xs)[0]) < loss


# --- shape and structure ---------------------------------------------------------------------


def test_gradient_tree_matches_the_network():
    net, xs = two_pairs(LinearLIF(tau=20.0, threshold=0.4)), drive()
    online, _ = eprop_grads(net, xs, squared)
    for layer, grad in zip(net.layers, online.layers, strict=True):
        assert type(layer) is type(grad)
    assert online.layers[0].weight.shape == net.layers[0].weight.shape


def test_bias_receives_a_gradient():
    net, xs = single_pair(LinearLIF(tau=20.0, threshold=0.4)), drive()
    online, _ = eprop_grads(net, xs, squared)
    assert online.layers[0].bias is not None
    assert jnp.any(online.layers[0].bias != 0)


def test_loss_value_matches_a_plain_forward_pass():
    net, xs = two_pairs(LinearLIF(tau=20.0, threshold=0.4)), drive()
    _, loss = eprop_grads(net, xs, squared)
    assert loss == pytest.approx(float(total_squared(unroll(net, xs)[0])), rel=1e-5)


def test_value_and_grad_wrapper_is_a_drop_in():
    net, xs = single_pair(LinearLIF(tau=20.0, threshold=0.4)), drive()
    step = eprop_value_and_grad(lambda output_t: squared(output_t))
    loss, grads = step(net, xs)
    direct_grads, direct_loss = eprop_grads(net, xs, squared)
    assert loss == pytest.approx(direct_loss)
    assert jnp.allclose(grads.layers[0].weight, direct_grads.layers[0].weight)


# --- memory ----------------------------------------------------------------------------------


def test_memory_grows_far_more_slowly_than_bptt():
    """The point of the whole exercise: traces are carried forward, not stored per timestep."""
    net = two_pairs(LinearLIF(tau=20.0, threshold=0.4))
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def temp_bytes(fn, xs):
        return jax.jit(fn).lower(params, xs).compile().memory_analysis().temp_size_in_bytes

    def bptt_fn(p, x):
        return eqx.filter_grad(lambda m, i: total_squared(unroll(m, i)[0]))(
            eqx.combine(p, static), x
        )

    def eprop_fn(p, x):
        return eprop_grads(eqx.combine(p, static), x, squared)[0]

    short, long = drive(t=100), drive(t=1000)
    bptt_growth = temp_bytes(bptt_fn, long) / temp_bytes(bptt_fn, short)
    eprop_growth = temp_bytes(eprop_fn, long) / temp_bytes(eprop_fn, short)
    assert bptt_growth > 5.0, f"BPTT should scale with T, grew {bptt_growth:.1f}x over 10x T"
    assert eprop_growth < 2.0, (
        f"e-prop should be near-flat in T, grew {eprop_growth:.1f}x against BPTT's "
        f"{bptt_growth:.1f}x"
    )


# --- refusals ---------------------------------------------------------------------------------


def test_recurrent_graph_is_refused():
    k = jax.random.split(jax.random.key(0), 3)
    net = Graph(
        nodes={"w": Dense(8, 8, key=k[0]), "n": LIF(tau=20.0), "r": Dense(8, 8, key=k[1])},
        edges=[("input", "w"), ("w", "n"), ("n", "r"), ("r", "n")],
        output="n",
    )
    with pytest.raises(TypeError, match="Recurrent Graphs are not supported"):
        eprop_grads(net, drive(features=8), squared)


def test_unsupported_neuron_is_refused_by_name():
    net = Sequential(Dense(12, 6, key=jax.random.key(0)), ALIF(tau=20.0))
    with pytest.raises(TypeError, match="ALIF"):
        eprop_grads(net, drive(), squared)


def test_odd_layer_count_is_refused():
    net = Sequential(Dense(12, 6, key=jax.random.key(0)))
    with pytest.raises(TypeError, match="even number"):
        eprop_grads(net, drive(), squared)
