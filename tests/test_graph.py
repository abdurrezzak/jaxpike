"""Graph topology tests.

The two rules that make an arbitrary wiring well-defined are that fan-in sums and that
cycle-closing edges read the previous timestep. Everything here checks one of those, or
checks that a malformed graph is rejected with a useful message.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxpike import (
    LIF,
    Dense,
    Graph,
    LeakyIntegrator,
    LinearLIF,
    Sequential,
    unroll,
    unroll_parallel,
)


def keys(n=6):
    return jax.random.split(jax.random.key(0), n)


def recurrent_net(hidden=16, features=8, classes=4):
    k = keys()
    return Graph(
        nodes={
            "w_in": Dense(features, hidden, key=k[0]),
            "hidden": LIF(tau=20.0, threshold=0.5),
            "w_rec": Dense(hidden, hidden, key=k[1]),
            "w_out": Dense(hidden, classes, key=k[2]),
            "out": LeakyIntegrator(tau=20.0),
        },
        edges=[
            ("input", "w_in"),
            ("w_in", "hidden"),
            ("hidden", "w_rec"),
            ("w_rec", "hidden"),
            ("hidden", "w_out"),
            ("w_out", "out"),
        ],
        output="out",
    )


def chain_as_graph(features=8, hidden=16):
    k = keys()
    return Graph(
        nodes={
            "w1": Dense(features, hidden, key=k[0]),
            "n1": LinearLIF(tau=20.0, threshold=0.5),
            "w2": Dense(hidden, hidden, key=k[1]),
            "n2": LinearLIF(tau=20.0, threshold=0.5),
        },
        edges=[("input", "w1"), ("w1", "n1"), ("n1", "w2"), ("w2", "n2")],
        output="n2",
    )


def drive(t=40, batch=3, features=8, seed=1, scale=4.0):
    return jax.random.normal(jax.random.key(seed), (t, batch, features)) * scale


# --- topology ------------------------------------------------------------------------------


def test_cycle_is_detected_and_marked_as_feedback():
    net = recurrent_net()
    assert net.is_recurrent
    assert net.back == frozenset({("w_rec", "hidden")})


def test_acyclic_graph_has_no_feedback():
    assert not chain_as_graph().is_recurrent


def test_topological_order_respects_forward_edges():
    order = list(recurrent_net().order)
    assert order.index("w_in") < order.index("hidden") < order.index("w_out")


def test_shapes_propagate_through_the_graph():
    shapes = recurrent_net().shapes((3, 8))
    assert shapes["hidden"] == (3, 16)
    assert shapes["out"] == (3, 4)
    assert recurrent_net().out_shape((3, 8)) == (3, 4)


# --- execution -----------------------------------------------------------------------------


def test_acyclic_graph_matches_the_equivalent_sequential():
    """A chain wired as a Graph must compute exactly what Sequential computes."""
    k = keys()
    graph = chain_as_graph()
    chain = Sequential(graph.nodes["w1"], graph.nodes["n1"], graph.nodes["w2"], graph.nodes["n2"])
    del k
    xs = drive()
    assert jnp.array_equal(unroll(graph, xs)[0], unroll(chain, xs)[0])


def test_recurrent_network_runs_and_differs_from_the_same_net_without_the_cycle():
    """The feedback edge must actually change the computation, not just exist."""
    net = recurrent_net()
    xs = drive()
    with_cycle, _ = unroll(net, xs)

    without = Graph(
        nodes=net.nodes,
        edges=[e for e in net.edges if e != ("w_rec", "hidden")],
        output="out",
    )
    no_cycle, _ = unroll(without, xs)
    assert jnp.all(jnp.isfinite(with_cycle))
    assert not jnp.allclose(with_cycle, no_cycle), "feedback had no effect"


def test_feedback_starts_from_zero_and_carries_across_chunks():
    net = recurrent_net()
    xs = drive(t=40)
    full, full_state = unroll(net, xs)
    first, mid = unroll(net, xs[:15])
    second, _ = unroll(net, xs[15:], mid)
    assert jnp.allclose(jnp.concatenate([first, second]), full, atol=1e-5)
    del full_state


def test_fan_in_sums_its_inputs():
    """Two edges into one node must add, which is what a synapse does."""
    k = keys()
    node_a = Dense(4, 4, key=k[0])
    node_b = Dense(4, 4, key=k[1])
    net = Graph(
        nodes={"a": node_a, "b": node_b, "n": LeakyIntegrator(tau=20.0)},
        edges=[("input", "a"), ("input", "b"), ("a", "n"), ("b", "n")],
        output="n",
    )
    xs = jnp.ones((1, 2, 4))
    out, _ = unroll(net, xs)

    solo_a = Graph(
        nodes={"a": node_a, "n": LeakyIntegrator(tau=20.0)},
        edges=[("input", "a"), ("a", "n")],
        output="n",
    )
    solo_b = Graph(
        nodes={"b": node_b, "n": LeakyIntegrator(tau=20.0)},
        edges=[("input", "b"), ("b", "n")],
        output="n",
    )
    expected = unroll(solo_a, xs)[0] + unroll(solo_b, xs)[0]
    assert jnp.allclose(out, expected, atol=1e-5)


def test_skip_connection_reaches_the_target():
    k = keys()
    net = Graph(
        nodes={
            "stem": Dense(8, 8, key=k[0]),
            "n1": LinearLIF(tau=20.0, threshold=0.3),
            "branch": Dense(8, 8, key=k[1]),
            "head": LeakyIntegrator(tau=20.0),
        },
        edges=[
            ("input", "stem"),
            ("stem", "n1"),
            ("n1", "branch"),
            ("branch", "head"),
            ("n1", "head"),  # the skip
        ],
        output="head",
    )
    assert net.out_shape((2, 8)) == (2, 8)
    out, _ = unroll(net, drive(features=8, batch=2))
    assert jnp.all(jnp.isfinite(out))


def test_gradients_flow_through_the_recurrent_weights():
    net = recurrent_net()
    xs = drive()

    def loss(module, inputs):
        return jnp.mean(unroll(module, inputs)[0])

    grads = eqx.filter_grad(loss)(net, xs)
    recurrent_grad = grads.nodes["w_rec"].weight
    assert jnp.all(jnp.isfinite(recurrent_grad))
    assert jnp.any(recurrent_grad != 0), "the feedback weight received no gradient"


# --- parallel-in-time ------------------------------------------------------------------------


def test_acyclic_graph_can_run_parallel_in_time():
    net = chain_as_graph()
    xs = drive()
    assert jnp.array_equal(unroll(net, xs)[0], unroll_parallel(net, xs)[0])


def test_recurrent_graph_refuses_parallel_in_time():
    with pytest.raises(TypeError, match="recurrent"):
        unroll_parallel(recurrent_net(), drive())


# --- rejections ------------------------------------------------------------------------------


def test_unknown_output_is_rejected():
    with pytest.raises(ValueError, match="is not a node"):
        Graph(nodes={"a": Dense(4, 4, key=keys()[0])}, edges=[("input", "a")], output="nope")


def test_unknown_edge_endpoint_is_rejected():
    with pytest.raises(ValueError, match="unknown target"):
        Graph(
            nodes={"a": Dense(4, 4, key=keys()[0])},
            edges=[("input", "a"), ("a", "ghost")],
            output="a",
        )


def test_graph_with_no_input_edge_is_rejected():
    with pytest.raises(ValueError, match="never fed"):
        Graph(nodes={"a": Dense(4, 4, key=keys()[0])}, edges=[("a", "a")], output="a")


def test_input_is_a_reserved_node_name():
    with pytest.raises(ValueError, match="reserved"):
        Graph(
            nodes={"input": Dense(4, 4, key=keys()[0])}, edges=[("input", "input")], output="input"
        )


def test_node_with_no_incoming_edge_is_rejected():
    """It would have nothing to compute from, and should fail at construction not at step 1."""
    k = keys()
    with pytest.raises(ValueError, match="no incoming edge"):
        Graph(
            nodes={"a": Dense(4, 4, key=k[0]), "orphan": Dense(4, 4, key=k[1])},
            edges=[("input", "a"), ("orphan", "a")],
            output="a",
        )


def test_subgraph_connected_only_through_its_own_cycle_is_rejected():
    k = keys()
    with pytest.raises(ValueError, match="unreachable"):
        Graph(
            nodes={
                "a": Dense(4, 4, key=k[0]),
                "x": Dense(4, 4, key=k[1]),
                "y": Dense(4, 4, key=k[2]),
            },
            edges=[("input", "a"), ("x", "y"), ("y", "x")],
            output="a",
        )


def test_mismatched_fan_in_shapes_are_rejected():
    k = keys()
    net = Graph(
        nodes={
            "a": Dense(4, 6, key=k[0]),
            "b": Dense(4, 8, key=k[1]),
            "n": LeakyIntegrator(tau=20.0),
        },
        edges=[("input", "a"), ("input", "b"), ("a", "n"), ("b", "n")],
        output="n",
    )
    with pytest.raises(ValueError, match="must share a shape"):
        net.out_shape((2, 4))
