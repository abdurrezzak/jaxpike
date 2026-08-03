"""Parallel-in-time execution tests.

The bar is that `unroll_parallel` is indistinguishable from `unroll` — outputs, final state,
and gradients — while removing the sequential dependency chain. A fast path that quietly
computes something slightly different is worse than no fast path.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxpike import LIF, Dense, LinearLIF, Sequential, unroll, unroll_parallel
from jaxpike.parallel import affine_compose, scan_linear_recurrence


def linear_net(key=None, threshold=0.2):
    # A low threshold on purpose. At the default 1.0 the second layer never fires -- layer 1
    # is sparse, so the deep layer's drive is tiny -- and the equivalence tests below would
    # then be comparing two all-zero arrays. Measured densities here: ~37% layer 1, ~10% out.
    k1, k2 = jax.random.split(key if key is not None else jax.random.key(0))
    return Sequential(
        Dense(8, 32, key=k1),
        LinearLIF(tau=20.0, threshold=threshold),
        Dense(32, 8, key=k2),
        LinearLIF(tau=20.0, threshold=threshold),
    )


def drive(t=128, batch=3, features=8, seed=1, scale=5.0):
    return jax.random.normal(jax.random.key(seed), (t, batch, features)) * scale


# --- equivalence -------------------------------------------------------------------------


def test_parallel_matches_sequential_for_a_single_neuron():
    neuron = LinearLIF(tau=20.0)
    xs = drive(features=16)
    seq, seq_state = unroll(neuron, xs)
    par, par_state = unroll_parallel(neuron, xs)
    assert jnp.array_equal(seq, par), "spike trains must be identical, not merely close"
    assert jnp.allclose(seq_state.v, par_state.v, atol=1e-5)


def test_parallel_matches_sequential_for_a_full_network():
    net, xs = linear_net(), drive()
    seq, seq_state = unroll(net, xs)
    par, par_state = unroll_parallel(net, xs)
    assert jnp.array_equal(seq, par)
    assert jnp.allclose(seq_state[3].v, par_state[3].v, atol=1e-5)


def test_the_test_drive_actually_produces_spikes():
    # Guards the equivalence tests above: two all-zero spike trains would match trivially.
    net, xs = linear_net(), drive()
    spikes, _ = unroll(net, xs)
    density = float(jnp.mean(spikes))
    assert density > 0.01, f"drive too weak ({density:.4f}); equivalence tests would be vacuous"
    assert density < 0.99, f"drive saturating ({density:.4f}); equally vacuous"


def test_parallel_matches_sequential_gradients():
    net, xs = linear_net(), drive()

    def loss(runner):
        return lambda m, x: jnp.mean(runner(m, x)[0])

    g_seq = eqx.filter_grad(loss(unroll))(net, xs)
    g_par = eqx.filter_grad(loss(unroll_parallel))(net, xs)
    leaves_s = jax.tree.leaves(eqx.filter(g_seq, eqx.is_inexact_array))
    leaves_p = jax.tree.leaves(eqx.filter(g_par, eqx.is_inexact_array))
    for a, b in zip(leaves_s, leaves_p, strict=True):
        assert jnp.allclose(a, b, atol=1e-4, rtol=1e-3)
    assert any(jnp.any(leaf != 0) for leaf in leaves_s)


@pytest.mark.parametrize("t", [1, 2, 3, 7, 64, 129])
def test_equivalence_holds_at_arbitrary_lengths(t):
    # Associative scans often break on non-power-of-two lengths; check the odd ones.
    neuron = LinearLIF(tau=20.0)
    xs = drive(t=t, features=4)
    assert jnp.array_equal(unroll(neuron, xs)[0], unroll_parallel(neuron, xs)[0])


def test_parallel_respects_a_nonzero_initial_state():
    neuron = LinearLIF(tau=20.0)
    xs = drive(features=4)
    first_seq, mid = unroll(neuron, xs[:40])
    first_par, mid_par = unroll_parallel(neuron, xs[:40])
    assert jnp.allclose(mid.v, mid_par.v, atol=1e-5)
    seq_rest, _ = unroll(neuron, xs[40:], mid)
    par_rest, _ = unroll_parallel(neuron, xs[40:], mid_par)
    assert jnp.array_equal(seq_rest, par_rest)
    assert jnp.array_equal(first_seq, first_par)


# --- dispatch and errors -----------------------------------------------------------------


def test_reset_based_neurons_are_rejected_by_name():
    net = Sequential(Dense(8, 16, key=jax.random.key(0)), LIF(tau=20.0))
    with pytest.raises(TypeError, match=r"layer 1 \(LIF\)"):
        unroll_parallel(net, drive())


def test_bare_reset_neuron_is_rejected_with_guidance():
    with pytest.raises(TypeError, match="LinearLIF"):
        unroll_parallel(LIF(tau=20.0), drive(features=4))


# --- the underlying primitive ------------------------------------------------------------


def test_affine_compose_is_associative():
    key = jax.random.key(3)
    e = [
        (jax.random.uniform(k, (5,)), jax.random.normal(k, (5,))) for k in jax.random.split(key, 3)
    ]
    left = affine_compose(affine_compose(e[0], e[1]), e[2])
    right = affine_compose(e[0], affine_compose(e[1], e[2]))
    for a, b in zip(left, right, strict=True):
        assert jnp.allclose(a, b, atol=1e-6)


def test_scan_linear_recurrence_matches_a_python_loop():
    t, n = 32, 6
    a = jnp.full((t, n), 0.9)
    b = jax.random.normal(jax.random.key(4), (t, n))
    v0 = jax.random.normal(jax.random.key(5), (n,))

    got = scan_linear_recurrence(a, b, v0)
    v, expected = v0, []
    for i in range(t):
        v = a[i] * v + b[i]
        expected.append(v)
    assert jnp.allclose(got, jnp.stack(expected), atol=1e-5)


def test_linear_lif_rejects_tau_below_dt():
    with pytest.raises(ValueError, match="tau must exceed dt"):
        LinearLIF(tau=0.5, dt=1.0)


def test_linear_lif_never_resets():
    """The defining property: membrane keeps integrating regardless of spiking."""
    neuron = LinearLIF(tau=20.0, threshold=0.5)
    xs = jnp.full((50, 1), 3.0)
    spikes, state = unroll(neuron, xs)
    assert jnp.sum(spikes) > 0
    assert state.v.item() > 0.5, "reset-free membrane must stay above threshold under drive"
