"""Initialization tests.

The property that matters is that a deep spiking network still fires at its output. This is
the SNN-specific failure mode: activity decays multiplicatively with depth and a silent
network has no gradient to recover from.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxpike import (
    Conv2d,
    Dense,
    Flatten,
    LinearLIF,
    Pool2d,
    Sequential,
    density,
    lif_gain,
    unroll,
)


def test_gain_grows_with_tau():
    """Longer time constants average over longer windows and need more compensation."""
    assert lif_gain(200.0) > lif_gain(20.0) > lif_gain(5.0) > 1.0


def test_gain_matches_the_closed_form():
    tau, dt = 20.0, 1.0
    a = jnp.exp(-dt / tau)
    assert lif_gain(tau, dt) == pytest.approx(float(jnp.sqrt((1 + a) / (1 - a))), rel=1e-6)
    assert lif_gain(20.0) == pytest.approx(6.325, abs=1e-3)


def test_gain_rejects_tau_below_dt():
    with pytest.raises(ValueError, match="tau must exceed dt"):
        lif_gain(0.5, 1.0)


def test_gain_restores_unit_membrane_variance():
    """The point of the factor: membrane std should land near 1, not 1/6."""
    tau = 20.0
    xs = jax.random.normal(jax.random.key(0), (4000, 1, 256))
    plain = LinearLIF(tau=tau, threshold=1e9)  # no spiking, just the filter

    _, state = unroll(plain, xs)
    del state

    # Feed the same noise scaled by the gain and measure the membrane's spread over time.
    def membrane_std(scale):
        neuron = LinearLIF(tau=tau, threshold=1e9)
        st = neuron.init_state((1, 256))
        vs = []
        for t in range(0, 4000, 200):
            st, _ = neuron(st, xs[t] * scale)
            vs.append(st.v)
        return float(jnp.std(jnp.stack(vs)))

    assert membrane_std(1.0) < 0.5, "unscaled input should under-drive the membrane"
    boosted = membrane_std(lif_gain(tau))
    assert 0.5 < boosted < 2.0, f"gain should bring membrane near unit scale, got {boosted}"


def deep_net(gain):
    k = jax.random.split(jax.random.key(0), 3)
    return Sequential(
        Conv2d(2, 8, 3, key=k[0], gain=gain),
        LinearLIF(tau=20.0, threshold=0.2),
        Pool2d(2),
        Conv2d(8, 16, 3, key=k[1], gain=gain),
        LinearLIF(tau=20.0, threshold=0.2),
        Pool2d(2),
        Flatten(),
        Dense(16 * 8 * 8, 10, key=k[2], gain=gain),
        LinearLIF(tau=20.0, threshold=0.2),
    )


def test_deep_network_is_silent_without_gain_and_alive_with_it():
    """The regression this whole module exists to prevent."""
    xs = jax.random.normal(jax.random.key(1), (12, 3, 32, 32, 2))
    silent, _ = unroll(deep_net(1.0), xs)
    alive, _ = unroll(deep_net(lif_gain(20.0)), xs)
    assert float(jnp.sum(silent)) == 0.0, "baseline should demonstrate the failure"
    assert float(density(alive)) > 0.05, "gain must keep the output layer firing"


def test_gain_keeps_firing_rate_stable_across_depth():
    xs = jax.random.normal(jax.random.key(1), (12, 3, 32, 32, 2))
    net = deep_net(lif_gain(20.0))
    rates = []
    for i in (1, 4, 8):
        out, _ = unroll(Sequential(*net.layers[: i + 1]), xs)
        rates.append(float(density(out)))
    assert all(r > 0.05 for r in rates), f"a layer went quiet: {rates}"
    # Some decay is expected; collapse is not.
    assert rates[-1] > rates[0] / 4, f"activity decayed too fast across depth: {rates}"
