"""Izhikevich neuron tests.

The model's whole claim is that four parameters reproduce cortex's qualitative firing
patterns, so the tests check the *patterns*, not just that numbers come out.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxpike import IZHIKEVICH_PRESETS, Izhikevich, unroll, unroll_parallel


def spike_times(spikes):
    return jnp.where(spikes[:, 0] > 0)[0]


def run(neuron, current=10.0, steps=400):
    return unroll(neuron, jnp.full((steps, 1), current))


@pytest.mark.parametrize("name", sorted(IZHIKEVICH_PRESETS))
def test_every_preset_fires_and_stays_finite(name):
    spikes, state = run(Izhikevich.preset(name))
    assert jnp.all((spikes == 0.0) | (spikes == 1.0))
    assert jnp.sum(spikes) > 0, f"{name} never fired"
    assert jnp.all(jnp.isfinite(state.v)), "the quadratic term must not run away"
    assert jnp.all(jnp.isfinite(state.u))


def test_unknown_preset_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown preset"):
        Izhikevich.preset("bursty_ish")


def test_fast_spiking_fires_faster_than_regular_spiking():
    """The defining contrast in Izhikevich (2003): FS lacks the adaptation RS has."""
    rs, _ = run(Izhikevich.preset("regular_spiking"))
    fs, _ = run(Izhikevich.preset("fast_spiking"))
    assert jnp.sum(fs) > jnp.sum(rs)


def test_chattering_fires_in_bursts_not_evenly():
    """Chattering neurons emit tight clusters, so their inter-spike intervals are irregular."""
    ch, _ = run(Izhikevich.preset("chattering"))
    fs, _ = run(Izhikevich.preset("fast_spiking"))
    ch_isi = jnp.diff(spike_times(ch)).astype(jnp.float32)
    fs_isi = jnp.diff(spike_times(fs)).astype(jnp.float32)
    assert jnp.std(ch_isi) > jnp.std(fs_isi), "bursting should be more irregular than tonic"


def test_stronger_current_produces_more_spikes():
    weak, _ = run(Izhikevich.preset("regular_spiking"), current=5.0)
    strong, _ = run(Izhikevich.preset("regular_spiking"), current=20.0)
    assert jnp.sum(strong) > jnp.sum(weak)


def test_no_current_means_no_spikes():
    spikes, _ = run(Izhikevich.preset("regular_spiking"), current=0.0)
    assert jnp.sum(spikes) == 0


def test_resets_to_c_after_a_spike():
    neuron = Izhikevich.preset("regular_spiking")
    # Step through manually and check the membrane lands near c on the step it fired.
    state = neuron.init_state((1,))
    xs = jnp.full((400, 1), 10.0)
    fired_at_least_once = False
    for t in range(120):
        state, s = neuron(state, xs[t])
        if float(s[0]) == 1.0:
            fired_at_least_once = True
            assert float(state.v[0]) == pytest.approx(neuron.c, abs=1.0)
    assert fired_at_least_once


def test_state_starts_at_rest():
    neuron = Izhikevich.preset("regular_spiking")
    state = neuron.init_state((4,))
    assert jnp.allclose(state.v, neuron.c)
    assert jnp.allclose(state.u, neuron.b * neuron.c)


def test_survives_extreme_input_without_nan():
    """The quadratic term is the failure mode; a huge current must clamp, not explode."""
    spikes, state = run(Izhikevich.preset("regular_spiking"), current=1e4, steps=100)
    assert jnp.all(jnp.isfinite(state.v)) and jnp.all(jnp.isfinite(state.u))
    assert jnp.all(jnp.isfinite(spikes))


def test_gradients_flow_through_the_surrogate():
    neuron = Izhikevich.preset("regular_spiking")

    def loss(current):
        spikes, _ = unroll(neuron, jnp.full((200, 1), current))
        return jnp.sum(spikes)

    g = jax.grad(loss)(10.0)
    assert jnp.isfinite(g)


def test_rejects_invalid_substeps():
    with pytest.raises(ValueError, match="substeps must be"):
        Izhikevich(substeps=0)


def test_is_not_parallelizable_and_says_so():
    """Nonlinear in time, so it must refuse rather than silently produce wrong results."""
    with pytest.raises(TypeError, match="parallel"):
        unroll_parallel(Izhikevich.preset("regular_spiking"), jnp.full((10, 1), 10.0))


def test_jit_matches_eager():
    neuron = Izhikevich.preset("chattering")
    xs = jnp.full((200, 1), 10.0)
    eager, _ = unroll(neuron, xs)
    compiled, _ = eqx.filter_jit(unroll)(neuron, xs)
    assert jnp.array_equal(eager, compiled)
