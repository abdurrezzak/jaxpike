"""Layer, container, and shape-propagation tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxpike import ALIF, LIF, Dense, Sequential, density, spike_rate, unroll


def make_net(in_features=8, hidden=16, out=4, key=None):
    k1, k2 = jax.random.split(key if key is not None else jax.random.key(0))
    return Sequential(
        Dense(in_features, hidden, key=k1),
        LIF(tau=20.0),
        Dense(hidden, out, key=k2),
        ALIF(tau=20.0),
    )


def test_shape_propagates_through_sequential():
    net = make_net()
    assert net.out_shape((5, 8)) == (5, 4)


def test_dense_rejects_mismatched_trailing_dim():
    layer = Dense(8, 4, key=jax.random.key(0))
    with pytest.raises(ValueError, match="trailing dim 8"):
        layer.out_shape((5, 7))


def test_state_structure_mirrors_layers_with_none_for_stateless():
    net = make_net()
    state = net.init_state((3, 8))
    assert len(state) == 4
    assert state[0] is None and state[2] is None, "Dense layers must be stateless"
    assert state[1].v.shape == (3, 16)
    assert state[3].v.shape == (3, 4) and state[3].a.shape == (3, 4)


def test_unroll_returns_time_major_output():
    net = make_net()
    xs = jax.random.normal(jax.random.key(1), (30, 3, 8))
    spikes, state = unroll(net, xs)
    assert spikes.shape == (30, 3, 4)
    assert state[3].v.shape == (3, 4)


def test_dense_matches_explicit_matmul():
    layer = Dense(6, 3, key=jax.random.key(2))
    x = jax.random.normal(jax.random.key(3), (5, 6))
    _, y = layer(None, x)
    assert jnp.allclose(y, x @ layer.weight.T + layer.bias, atol=1e-6)


def test_dense_without_bias_has_no_bias_leaf():
    layer = Dense(6, 3, key=jax.random.key(4), use_bias=False)
    assert layer.bias is None


def test_readouts_are_consistent_with_each_other():
    net = make_net()
    xs = jax.random.normal(jax.random.key(5), (40, 3, 8)) * 3.0
    spikes, _ = unroll(net, xs)
    assert jnp.allclose(spike_rate(spikes), jnp.mean(spikes, axis=0))
    assert 0.0 <= density(spikes) <= 1.0


@settings(deadline=None, max_examples=15)
@given(
    t=st.integers(min_value=1, max_value=12),
    batch=st.integers(min_value=1, max_value=5),
    features=st.integers(min_value=1, max_value=9),
)
def test_arbitrary_shapes_round_trip(t, batch, features):
    net = Sequential(Dense(features, 7, key=jax.random.key(6)), LIF(tau=20.0))
    xs = jnp.ones((t, batch, features))
    spikes, _ = unroll(net, xs)
    assert spikes.shape == (t, batch, 7)
    assert jnp.all((spikes == 0.0) | (spikes == 1.0))
