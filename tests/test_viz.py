"""Visualization tests.

Plots cannot be asserted on visually, so these check the contract instead: correct return
type, accepted input shapes, refusals on malformed input, and — the parts that carry meaning
— that the diverging scale is symmetric about zero and that `layer_rates` flags a silent
network. Rendering happens on the Agg backend so this runs headless in CI.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jaxpike import (  # noqa: E402
    LIF,
    Dense,
    LinearLIF,
    Sequential,
    lif_gain,
    unroll,
)
from jaxpike.viz import (  # noqa: E402
    Theme,
    layer_rates,
    layer_rates_from,
    membrane,
    raster,
    rate_heatmap,
    weights,
)

THEMES = [Theme.light(), Theme.dark()]
IDS = ["light", "dark"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def spike_train(t=60, neurons=20, seed=0, rate=0.15):
    return (jax.random.uniform(jax.random.key(seed), (t, neurons)) < rate).astype(jnp.float32)


def net(threshold=0.9):
    gain = lif_gain(20.0)
    k = jax.random.split(jax.random.key(0), 2)
    return Sequential(
        Dense(30, 40, key=k[0], gain=gain),
        LinearLIF(tau=20.0, threshold=threshold),
        Dense(40, 10, key=k[1], gain=gain),
        LinearLIF(tau=20.0, threshold=threshold),
    )


# --- contract ------------------------------------------------------------------------------


@pytest.mark.parametrize("theme", THEMES, ids=IDS)
def test_every_plot_returns_an_axes(theme):
    spikes = spike_train()
    assert isinstance(raster(spikes, theme=theme), plt.Axes)
    assert isinstance(membrane(spikes, threshold=1.0, theme=theme), plt.Axes)
    assert isinstance(layer_rates([0.1, 0.2], theme=theme), plt.Axes)
    assert isinstance(rate_heatmap(spikes, theme=theme), plt.Axes)
    assert isinstance(weights(jax.random.normal(jax.random.key(1), (8, 6)), theme=theme), plt.Axes)


def test_plots_compose_into_a_supplied_axes():
    fig, axes = plt.subplots(1, 2)
    assert raster(spike_train(), ax=axes[0]) is axes[0]
    assert rate_heatmap(spike_train(), ax=axes[1]) is axes[1]
    assert len(fig.axes) >= 2


@pytest.mark.parametrize("shape", [(40, 12), (40, 3, 12)], ids=["TN", "TBN"])
def test_batched_and_unbatched_inputs_both_work(shape):
    data = (jax.random.uniform(jax.random.key(0), shape) < 0.2).astype(jnp.float32)
    assert raster(data) is not None
    assert rate_heatmap(data) is not None


def test_rejects_wrongly_shaped_spikes():
    with pytest.raises(ValueError, match=r"\(T, N\) or \(T, B, N\)"):
        raster(jnp.zeros((4, 4, 4, 4)))


def test_weights_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        weights(jnp.zeros((3, 3, 3)))


def test_raster_switches_to_image_above_max_neurons():
    """Individual markers stop resolving, so the dense path must engage rather than overplot."""
    dense = (jax.random.uniform(jax.random.key(0), (30, 400)) < 0.1).astype(jnp.float32)
    ax = raster(dense, max_neurons=100)
    assert ax.images and not ax.collections
    sparse = raster(spike_train(neurons=20), max_neurons=100)
    assert sparse.collections and not sparse.images


# --- meaning -------------------------------------------------------------------------------


def test_weight_scale_is_symmetric_about_zero():
    """Asymmetric limits would put zero at an arbitrary colour and imply a bias that isn't there."""
    skewed = jnp.array([[-0.2, 0.1], [3.0, 0.4]])
    ax = weights(skewed)
    vmin, vmax = ax.images[0].get_clim()
    assert vmin == pytest.approx(-vmax)
    assert vmax == pytest.approx(3.0)


def test_weights_of_all_zeros_does_not_divide_by_zero():
    ax = weights(jnp.zeros((4, 4)))
    vmin, vmax = ax.images[0].get_clim()
    assert vmin < vmax


def test_layer_rates_flags_a_silent_layer():
    ax = layer_rates([0.3, 0.05, 0.0], labels=["a", "b", "c"])
    texts = [t.get_text() for t in ax.texts]
    assert "silent" in texts, f"a zero-rate layer must be called out, got {texts}"


def test_layer_rates_flags_a_saturating_layer():
    ax = layer_rates([0.3, 0.99], healthy=(0.02, 0.5))
    assert "saturating" in [t.get_text() for t in ax.texts]


def test_layer_rates_leaves_a_healthy_network_unflagged():
    ax = layer_rates([0.3, 0.25, 0.2], healthy=(0.02, 0.5))
    texts = [t.get_text() for t in ax.texts]
    assert "silent" not in texts and "saturating" not in texts


def test_layer_rates_from_measures_every_spiking_layer():
    xs = jax.random.normal(jax.random.key(3), (40, 1, 30))
    ax = layer_rates_from(net(), xs)
    # ax.patches also holds the shaded healthy band, so count the bar container instead.
    assert len(ax.containers[0]) == 2, "one bar per spiking layer, Dense layers excluded"


def test_layer_rates_from_refuses_a_network_with_no_spiking_layers():
    xs = jax.random.normal(jax.random.key(3), (10, 1, 30))
    with pytest.raises(ValueError, match="no spiking layers"):
        layer_rates_from(Sequential(Dense(30, 10, key=jax.random.key(0))), xs)


def test_membrane_marks_the_spikes_it_is_given():
    lif = LIF(tau=20.0, threshold=1.0)
    drive = jnp.full((80, 1), 2.5)
    spikes, _ = unroll(lif, drive)
    state, trace = lif.init_state((1,)), []
    for t in range(80):
        state, _ = lif(state, drive[t])
        trace.append(state.v)
    ax = membrane(jnp.stack(trace), spikes=spikes, threshold=1.0)
    assert jnp.sum(spikes) > 0
    assert ax.collections, "spike markers should be drawn"
    assert ax.get_legend() is not None


def test_rate_heatmap_binning_is_clamped_to_sequence_length():
    ax = rate_heatmap(spike_train(t=10), bins=500)
    assert ax.images[0].get_array().shape[1] == 10


def test_themes_differ_and_dark_is_not_an_inversion():
    light, dark = Theme.light(), Theme.dark()
    assert light.surface != dark.surface
    assert light.series != dark.series, "dark steps are selected, not flipped"
    assert len(light.series) == len(dark.series) == 8
