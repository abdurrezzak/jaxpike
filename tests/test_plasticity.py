"""STDP tests.

The property that defines STDP is causality: pre-before-post strengthens, post-before-pre
weakens, and the effect decays with the time gap. Everything else is bookkeeping.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxpike import STDP, stdp_window

T, W0 = 60, 0.5


def pair(pre_t, post_t, *, rule=None, w0=W0, n=1):
    rule = rule or STDP()
    pre = jnp.zeros((T, 1, n)).at[pre_t, 0, 0].set(1.0)
    post = jnp.zeros((T, 1, n)).at[post_t, 0, 0].set(1.0)
    new, _ = rule(jnp.full((n, n), w0), pre, post)
    return float(new[0, 0] - w0)


def test_pre_before_post_strengthens():
    assert pair(10, 15) > 0


def test_post_before_pre_weakens():
    assert pair(15, 10) < 0


def test_simultaneous_spikes_do_nothing():
    """Same timestep is not a causal order, so neither direction should be credited."""
    assert pair(10, 10) == pytest.approx(0.0, abs=1e-9)


def test_effect_decays_with_timing_gap():
    close = pair(10, 12)
    far = pair(10, 40)
    assert close > far > 0


def test_depression_outweighs_potentiation_at_equal_timing():
    """Default a_minus > a_plus, the standard asymmetry that stops runaway strengthening."""
    assert abs(pair(15, 10)) > abs(pair(10, 15))


def test_weights_are_clamped_to_bounds():
    rule = STDP(a_plus=10.0, a_minus=10.0, w_min=0.0, w_max=1.0)
    # Compare the resulting weights, not deltas: reconstructing w0 + delta in float32
    # introduces its own rounding and would test arithmetic rather than the clamp.
    strong = 0.9 + pair(10, 15, rule=rule, w0=0.9)
    weak = 0.1 + pair(15, 10, rule=rule, w0=0.1)
    assert strong == pytest.approx(1.0, abs=1e-6)
    assert weak == pytest.approx(0.0, abs=1e-6)


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="must be below"):
        STDP(w_min=1.0, w_max=0.0)


def test_rejects_mismatched_sequence_lengths():
    rule = STDP()
    with pytest.raises(ValueError, match="same timesteps"):
        rule(jnp.zeros((2, 3)), jnp.zeros((10, 1, 3)), jnp.zeros((8, 1, 2)))


def test_shapes_and_batching():
    rule = STDP()
    pre = (jax.random.uniform(jax.random.key(0), (T, 4, 6)) < 0.1).astype(jnp.float32)
    post = (jax.random.uniform(jax.random.key(1), (T, 4, 3)) < 0.1).astype(jnp.float32)
    w = jnp.full((3, 6), 0.5)
    new, traces = rule(w, pre, post)
    assert new.shape == (3, 6)
    assert traces.pre.shape == (4, 6) and traces.post.shape == (4, 3)


def test_traces_carry_across_calls():
    """Streaming: splitting a sequence and passing traces through matches one call."""
    rule = STDP()
    pre = (jax.random.uniform(jax.random.key(0), (T, 2, 4)) < 0.15).astype(jnp.float32)
    post = (jax.random.uniform(jax.random.key(1), (T, 2, 4)) < 0.15).astype(jnp.float32)
    w = jnp.full((4, 4), 0.5)

    whole, _ = rule(w, pre, post)
    part, traces = rule(w, pre[:30], post[:30])
    rest, _ = rule(part, pre[30:], post[30:], state=traces)
    assert jnp.allclose(whole, rest, atol=1e-6)


def test_no_spikes_means_no_change():
    rule = STDP()
    w = jnp.full((3, 3), 0.5)
    new, _ = rule(w, jnp.zeros((T, 1, 3)), jnp.zeros((T, 1, 3)))
    assert jnp.allclose(new, w)


def test_window_has_the_classic_asymmetric_shape():
    dt = jnp.array([-40.0, -20.0, -5.0, 0.0, 5.0, 20.0, 40.0])
    w = stdp_window(dt)
    assert jnp.all(w[:3] < 0), "negative delta_t (post before pre) must depress"
    assert w[3] == 0.0, "zero delta_t is undefined ordering"
    assert jnp.all(w[4:] > 0), "positive delta_t (pre before post) must potentiate"
    assert w[4] > w[5] > w[6], "potentiation decays with the gap"


def test_window_matches_the_pairwise_simulation_in_sign():
    for gap in (2, 5, 10, 25):
        assert jnp.sign(stdp_window(jnp.array(float(gap)))) == jnp.sign(pair(10, 10 + gap))
        assert jnp.sign(stdp_window(jnp.array(float(-gap)))) == jnp.sign(pair(10 + gap, 10))
