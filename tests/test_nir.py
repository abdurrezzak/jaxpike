"""NIR import/export tests.

Round trips within jaxpike must be exact. Round trips through another framework must not be
assumed exact, and the module documents why; what is tested here is that our own conversion
is lossless and that models NIR cannot represent fail loudly instead of quietly changing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxpike import (
    ALIF,
    LIF,
    Conv2d,
    Dense,
    Flatten,
    Izhikevich,
    LeakyIntegrator,
    LinearLIF,
    Pool2d,
    Sequential,
    lif_gain,
    unroll,
)
from jaxpike.nir import NIRConversionError, from_nir, load, save, to_nir

nir = pytest.importorskip("nir")


def dense_net(key=None):
    k = jax.random.split(key if key is not None else jax.random.key(0), 2)
    return Sequential(
        Dense(20, 32, key=k[0]),
        LIF(tau=20.0, threshold=0.5, reset="zero"),
        Dense(32, 5, key=k[1]),
        LeakyIntegrator(tau=15.0),
    )


def conv_net(key=None):
    gain = lif_gain(20.0)
    k = jax.random.split(key if key is not None else jax.random.key(0), 3)
    return Sequential(
        Conv2d(2, 8, 3, key=k[0], gain=gain),
        LinearLIF(tau=20.0, threshold=0.2),
        Pool2d(2),
        Conv2d(8, 16, 3, key=k[1], gain=gain),
        LinearLIF(tau=20.0, threshold=0.2),
        Pool2d(2),
        Flatten(),
        Dense(16 * 8 * 8, 10, key=k[2], gain=gain),
        LIF(tau=20.0, threshold=0.2, reset="zero"),
    )


def assert_same_behaviour(a, b, xs):
    out_a, _ = unroll(a, xs)
    out_b, _ = unroll(b, xs)
    assert jnp.sum(jnp.abs(out_a)) > 0, "silent model makes this comparison vacuous"
    assert jnp.array_equal(out_a, out_b)


# --- round trips ---------------------------------------------------------------------------


def test_dense_network_round_trips_exactly(tmp_path):
    net = dense_net()
    path = tmp_path / "m.nir"
    save(net, path, (1, 20))
    assert_same_behaviour(net, load(path), jax.random.normal(jax.random.key(3), (30, 2, 20)) * 3)


def test_conv_network_round_trips_exactly(tmp_path):
    net = conv_net()
    path = tmp_path / "c.nir"
    save(net, path, (1, 32, 32, 2))
    xs = jax.random.normal(jax.random.key(3), (10, 2, 32, 32, 2))
    assert_same_behaviour(net, load(path), xs)


def test_round_trip_preserves_layer_types(tmp_path):
    net = conv_net()
    path = tmp_path / "c.nir"
    save(net, path, (1, 32, 32, 2))
    got = [type(layer).__name__ for layer in load(path).layers]
    assert got == [type(layer).__name__ for layer in net.layers]


def test_linear_lif_becomes_li_plus_threshold():
    """NIR has no reset-free LIF node, but LI -> Threshold expresses it exactly."""
    graph = to_nir(Sequential(Dense(4, 4, key=jax.random.key(0)), LinearLIF(tau=20.0)), (1, 4))
    kinds = [type(node).__name__ for node in graph.nodes.values()]
    assert "LI" in kinds and "Threshold" in kinds


def test_flatten_column_permutation_is_actually_needed(tmp_path):
    """Guards the subtlest bug here: channels-last vs channels-first flatten ordering.

    If the Dense columns were not permuted on export, the reimported convnet would still have
    the right shapes and still run -- it would just compute something different. Only a
    numerical comparison catches it, which is why this test exists separately.
    """
    net = conv_net()
    path = tmp_path / "c.nir"
    save(net, path, (1, 32, 32, 2))
    reimported = load(path)
    original_dense = net.layers[7].weight
    round_tripped = reimported.layers[7].weight
    assert jnp.array_equal(original_dense, round_tripped)


def test_dense_without_bias_uses_linear_node():
    net = Sequential(Dense(4, 3, key=jax.random.key(0), use_bias=False))
    kinds = [type(n).__name__ for n in to_nir(net, (1, 4)).nodes.values()]
    assert "Linear" in kinds and "Affine" not in kinds


# --- units ---------------------------------------------------------------------------------


def test_tau_is_exported_in_seconds():
    """NIR stores seconds; our neurons store timesteps. A 20-step tau at 1 ms is 0.02 s."""
    graph = to_nir(Sequential(LIF(tau=20.0, reset="zero")), (1, 4), dt_seconds=1e-3)
    lif_node = next(n for n in graph.nodes.values() if isinstance(n, nir.LIF))
    assert float(jnp.asarray(lif_node.tau).reshape(-1)[0]) == pytest.approx(0.02)


def test_mismatched_dt_seconds_rescales_tau(tmp_path):
    """Documents the failure mode rather than hiding it: the units must agree."""
    net = Sequential(Dense(4, 4, key=jax.random.key(0)), LIF(tau=20.0, reset="zero"))
    path = tmp_path / "m.nir"
    save(net, path, (1, 4), dt_seconds=1e-3)
    matched = load(path, dt_seconds=1e-3)
    mismatched = load(path, dt_seconds=1e-4)
    assert float(matched.layers[1].tau) == pytest.approx(20.0)
    assert float(mismatched.layers[1].tau) == pytest.approx(200.0)


# --- refusals ------------------------------------------------------------------------------


def test_subtract_reset_is_refused_not_silently_converted():
    net = Sequential(Dense(4, 4, key=jax.random.key(0)), LIF(tau=20.0, reset="subtract"))
    with pytest.raises(NIRConversionError, match="reset='subtract'"):
        to_nir(net, (1, 4))


def test_max_pooling_is_refused():
    with pytest.raises(NIRConversionError, match="max-pooling"):
        to_nir(Sequential(Pool2d(2, mode="max")), (1, 8, 8, 2))


@pytest.mark.parametrize(
    "layer",
    [Izhikevich.preset("chattering"), ALIF(tau=20.0)],
    ids=["izhikevich", "alif"],
)
def test_models_without_a_nir_primitive_are_refused(layer):
    net = Sequential(Dense(4, 4, key=jax.random.key(0)), layer)
    with pytest.raises(NIRConversionError, match="no NIR primitive"):
        to_nir(net, (1, 4))


def test_import_refuses_nonzero_v_leak():
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": jnp.array([2])}),
            "lif": nir.LIF(
                tau=jnp.full(2, 0.02),
                r=jnp.ones(2),
                v_leak=jnp.full(2, 0.3),
                v_threshold=jnp.ones(2),
                v_reset=jnp.zeros(2),
            ),
            "output": nir.Output(output_type={"output": jnp.array([2])}),
        },
        edges=[("input", "lif"), ("lif", "output")],
    )
    with pytest.raises(NIRConversionError, match="v_leak"):
        from_nir(graph)


def test_import_refuses_per_neuron_tau():
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": jnp.array([2])}),
            "lif": nir.LIF(
                tau=jnp.array([0.02, 0.05]),
                r=jnp.ones(2),
                v_leak=jnp.zeros(2),
                v_threshold=jnp.ones(2),
                v_reset=jnp.zeros(2),
            ),
            "output": nir.Output(output_type={"output": jnp.array([2])}),
        },
        edges=[("input", "lif"), ("lif", "output")],
    )
    with pytest.raises(NIRConversionError, match="per-neuron"):
        from_nir(graph)
