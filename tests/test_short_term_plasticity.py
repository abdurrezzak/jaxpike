"""Tsodyks-Markram short-term plasticity and Izhikevich's dopamine-modulated STDP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxpike import (
    MARKRAM_PRESETS,
    DopamineSTDP,
    TsodyksMarkram,
    unroll,
    unroll_parallel,
)


def train(rate_hz=20, steps=400, dt=1.0):
    """Regular spike train at `rate_hz`, with dt in milliseconds."""
    period = int(1000 / rate_hz / dt)
    return jnp.zeros((steps, 1)).at[jnp.arange(0, steps, period), 0].set(1.0)


def amplitudes(syn, spikes):
    out, _ = unroll(syn, spikes)
    return out[out[:, 0] > 0][:, 0]


# --- Tsodyks-Markram ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MARKRAM_PRESETS))
def test_every_preset_transmits_and_stays_bounded(name):
    amps = amplitudes(TsodyksMarkram.preset(name), train())
    assert len(amps) > 3
    assert jnp.all(amps > 0.0) and jnp.all(amps <= 1.0)
    assert jnp.all(jnp.isfinite(amps))


def test_depressing_synapses_weaken_across_a_train():
    amps = amplitudes(TsodyksMarkram.preset("depressing"), train())
    assert amps[0] > amps[1] > amps[2], "resources should deplete spike by spike"
    assert float(amps[-1]) < float(amps[0]) / 2


def test_facilitating_synapses_strengthen_across_a_train():
    amps = amplitudes(TsodyksMarkram.preset("facilitating"), train())
    assert amps[0] < amps[1] < amps[2], "utilization should build spike by spike"


def test_first_spike_amplitude_is_the_baseline_release_probability():
    """Before any history, u jumps to U and x is still 1, so the first release is exactly U."""
    for name, (u, _, _) in MARKRAM_PRESETS.items():
        amps = amplitudes(TsodyksMarkram.preset(name), train())
        assert float(amps[0]) == pytest.approx(u, rel=1e-4), name


def test_recovery_between_widely_spaced_spikes():
    """Two spikes far apart should transmit nearly identically -- resources have recovered."""
    syn = TsodyksMarkram.preset("depressing")
    spikes = jnp.zeros((4000, 1)).at[jnp.array([10, 3500]), 0].set(1.0)
    amps = amplitudes(syn, spikes)
    assert float(amps[1]) == pytest.approx(float(amps[0]), rel=0.15)


def test_higher_rates_depress_more():
    syn = TsodyksMarkram.preset("depressing")
    slow = amplitudes(syn, train(rate_hz=5))
    fast = amplitudes(syn, train(rate_hz=50))
    assert float(fast[-1]) < float(slow[-1]), "depression must be rate-dependent"


def test_no_spikes_transmit_nothing():
    out, _ = unroll(TsodyksMarkram.preset("depressing"), jnp.zeros((100, 3)))
    assert jnp.all(out == 0.0)


def test_parallel_apply_matches_sequential():
    for name in ("depressing", "facilitating", "F3_mixed"):
        syn = TsodyksMarkram.preset(name)
        spikes = (jax.random.uniform(jax.random.key(0), (300, 2, 5)) < 0.05).astype(jnp.float32)
        seq, seq_state = unroll(syn, spikes)
        par, par_state = unroll_parallel(syn, spikes)
        assert jnp.allclose(seq, par, atol=1e-5), name
        assert jnp.allclose(seq_state.x, par_state.x, atol=1e-5)
        assert jnp.allclose(seq_state.u, par_state.u, atol=1e-5)


def test_zero_tau_f_disables_facilitation():
    syn = TsodyksMarkram(U=0.5, tau_d=800.0, tau_f=0.0)
    amps = amplitudes(syn, train())
    assert float(amps[0]) == pytest.approx(0.5, rel=1e-4)
    assert amps[1] < amps[0], "with no facilitation the synapse can only depress"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"U": 0.0}, "release probability"),
        ({"U": 1.5}, "release probability"),
        ({"tau_d": 0.0}, "tau_d must be positive"),
        ({"tau_f": -1.0}, "tau_f must be non-negative"),
    ],
)
def test_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TsodyksMarkram(**kwargs)


def test_unknown_preset_lists_options():
    with pytest.raises(ValueError, match="unknown preset"):
        TsodyksMarkram.preset("sticky")


# --- dopamine-modulated STDP --------------------------------------------------------------


def paired(delay_ms, *, steps=2500, rule=None, pre_t=100, gap=5, w0=0.5):
    """One causal spike pair, with reward `delay_ms` later. Returns the weight change."""
    rule = rule or DopamineSTDP()
    pre = jnp.zeros((steps, 1, 1)).at[pre_t, 0, 0].set(1.0)
    post = jnp.zeros((steps, 1, 1)).at[pre_t + gap, 0, 0].set(1.0)
    reward = jnp.zeros(steps)
    if delay_ms is not None:
        reward = reward.at[pre_t + delay_ms].set(1.0)
    new, _ = rule(jnp.array([[w0]]), pre, post, reward)
    return float(new[0, 0] - w0)


def test_no_reward_means_no_weight_change():
    """The defining property: STDP alone writes only to the eligibility trace."""
    assert paired(None) == pytest.approx(0.0, abs=1e-12)


def test_reward_seconds_later_still_reinforces():
    """The distal reward problem. A reward 2 s after the spike pair must still act."""
    assert paired(2000) > 0.0


def test_effect_decays_with_reward_delay():
    immediate, medium, late = paired(0), paired(500), paired(2000)
    assert immediate > medium > late > 0.0


def test_anticausal_pairing_is_punished_by_reward():
    """Reward amplifies whatever the eligibility trace holds, including negative tags."""
    rule = DopamineSTDP()
    pre = jnp.zeros((1500, 1, 1)).at[105, 0, 0].set(1.0)
    post = jnp.zeros((1500, 1, 1)).at[100, 0, 0].set(1.0)  # post before pre
    reward = jnp.zeros(1500).at[300].set(1.0)
    new, _ = rule(jnp.array([[0.5]]), pre, post, reward)
    assert float(new[0, 0] - 0.5) < 0.0


def test_reward_without_any_spikes_does_nothing():
    rule = DopamineSTDP()
    reward = jnp.zeros(500).at[100].set(1.0)
    new, _ = rule(jnp.full((2, 3), 0.5), jnp.zeros((500, 1, 3)), jnp.zeros((500, 1, 2)), reward)
    assert jnp.allclose(new, 0.5), "dopamine with no eligibility must be inert"


def test_weights_stay_within_bounds():
    rule = DopamineSTDP(dopamine_per_reward=1e3, w_min=0.0, w_max=1.0)
    assert 0.9 + paired(50, rule=rule, w0=0.9) == pytest.approx(1.0, abs=1e-6)


def test_rejects_mismatched_lengths():
    rule = DopamineSTDP()
    with pytest.raises(ValueError, match="same timesteps"):
        rule(jnp.zeros((1, 1)), jnp.zeros((10, 1, 1)), jnp.zeros((10, 1, 1)), jnp.zeros(7))


def test_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="must be below"):
        DopamineSTDP(w_min=1.0, w_max=0.5)


def test_state_carries_across_calls():
    rule = DopamineSTDP()
    pre = (jax.random.uniform(jax.random.key(0), (400, 1, 3)) < 0.05).astype(jnp.float32)
    post = (jax.random.uniform(jax.random.key(1), (400, 1, 2)) < 0.05).astype(jnp.float32)
    reward = jnp.zeros(400).at[200].set(1.0)
    w = jnp.full((2, 3), 0.5)

    whole, _ = rule(w, pre, post, reward)
    part, state = rule(w, pre[:150], post[:150], reward[:150])
    rest, _ = rule(part, pre[150:], post[150:], reward[150:], state=state)
    assert jnp.allclose(whole, rest, atol=1e-6)
