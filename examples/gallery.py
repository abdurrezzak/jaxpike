"""Render the visualization gallery into docs/figures/.

Uses real Spiking Heidelberg Digits data where possible — a raster of an actual spoken digit
says far more about what these plots are for than random noise does. Falls back to synthetic
input if the dataset is not cached.

    PYTHONPATH=. python examples/gallery.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jaxpike as jp
from jaxpike import viz

OUT = Path("docs/figures")
TAU, THRESHOLD = 20.0, 0.6


def load_sample(timesteps: int = 250):
    """One SHD utterance as (T, channels), or synthetic input if the data is not present."""
    try:
        from examples.shd import load

        xs, ys = load("test", Path("data/shd"), timesteps=timesteps)
        return jnp.asarray(xs[0], jnp.float32), int(ys[0]), True
    except Exception:
        key = jax.random.key(0)
        synthetic = (jax.random.uniform(key, (timesteps, 700)) < 0.02).astype(jnp.float32)
        return synthetic, -1, False


def build(features: int, key):
    gain = jp.lif_gain(TAU)
    k1, k2, k3 = jax.random.split(key, 3)
    return jp.Sequential(
        jp.Dense(features, 128, key=k1, gain=gain),
        jp.LinearLIF(tau=TAU, threshold=THRESHOLD),
        jp.Dense(128, 64, key=k2, gain=gain),
        jp.LinearLIF(tau=TAU, threshold=THRESHOLD),
        jp.Dense(64, 20, key=k3, gain=gain),
        jp.LinearLIF(tau=TAU, threshold=THRESHOLD),
    )


def lif_trace(steps=200, current=1.8):
    """Membrane trace of a single LIF under constant drive, plus its spikes."""
    lif = jp.LIF(tau=TAU, threshold=1.0)
    xs = jnp.full((steps, 1), current)
    spikes, _ = jp.unroll(lif, xs)
    state, trace = lif.init_state((1,)), []
    for t in range(steps):
        state, _ = lif(state, xs[t])
        trace.append(state.v)
    return jnp.stack(trace), spikes


def gallery(theme: viz.Theme, sample, net, label: str, real: bool) -> plt.Figure:
    hidden, _ = jp.unroll(jp.Sequential(*net.layers[:2]), sample[:, None, :])
    voltage, spikes = lif_trace()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.patch.set_facecolor(theme.surface)

    viz.raster(sample, ax=axes[0, 0], theme=theme, title=f"Input spikes — {label}")
    viz.raster(hidden, ax=axes[0, 1], theme=theme, title="Hidden layer 1 response")
    viz.membrane(voltage, spikes=spikes, threshold=1.0, ax=axes[1, 0], theme=theme)
    viz.layer_rates_from(net, sample[:, None, :], ax=axes[1, 1], theme=theme)
    viz.rate_heatmap(hidden, ax=axes[2, 0], theme=theme, title="Hidden firing rate over time")
    viz.weights(
        net.layers[2].weight[:64, :64],
        ax=axes[2, 1],
        theme=theme,
        title="Weights — hidden 1 to hidden 2",
    )

    source = "real SHD data" if real else "synthetic input"
    fig.suptitle(
        f"jaxpike visualization gallery — {source}",
        x=0.006,
        ha="left",
        fontsize=13,
        color=theme.text,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def izhikevich_panel(theme: viz.Theme) -> plt.Figure:
    """Every Izhikevich preset under identical drive — the model's whole selling point."""
    names = list(jp.IZHIKEVICH_PRESETS)
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.55 * len(names)), sharex=True)
    fig.patch.set_facecolor(theme.surface)

    steps, current = 320, 10.0
    for ax, name in zip(axes, names, strict=True):
        neuron = jp.Izhikevich.preset(name)
        xs = jnp.full((steps, 1), current)
        state, trace = neuron.init_state((1,)), []
        spikes = []
        for t in range(steps):
            state, s = neuron(state, xs[t])
            trace.append(float(state.v[0]))
            spikes.append(float(s[0]))
        viz._prepare(ax, theme, None)
        ax.plot(trace, linewidth=1.8, color=theme.series[0], solid_capstyle="round")
        fired = [t for t, s in enumerate(spikes) if s > 0]
        ax.scatter(
            fired,
            [neuron.v_peak + 22] * len(fired),
            s=18,
            marker="v",
            color=theme.series[1],
            zorder=3,
        )
        ax.set_ylabel("mV", fontsize=9)
        ax.set_ylim(-95, 78)
        ax.text(
            0.004,
            0.99,
            name.replace("_", " "),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=theme.text,
            va="top",
        )
        ax.text(
            0.999,
            0.99,
            f"{len(fired)} spikes",
            transform=ax.transAxes,
            fontsize=9,
            color=theme.secondary,
            ha="right",
            va="top",
        )
    axes[-1].set_xlabel("timestep")
    axes[-1].set_xlim(0, steps)
    fig.suptitle(
        f"Izhikevich firing patterns — identical drive (I = {current:g})",
        x=0.006,
        ha="left",
        fontsize=13,
        color=theme.text,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plasticity_panel(theme: viz.Theme) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    fig.patch.set_facecolor(theme.surface)

    # 1. The classic STDP window.
    ax = viz._prepare(axes[0], theme, None)
    dt = jnp.linspace(-80, 80, 400)
    window = jp.stdp_window(dt)
    ax.axhline(0, color=theme.grid, linewidth=1.2)
    ax.axvline(0, color=theme.grid, linewidth=1.2)
    ax.plot(dt, window, linewidth=2.2, color=theme.series[0], solid_capstyle="round")
    ax.fill_between(dt, window, 0, where=(window > 0), color=theme.series[0], alpha=0.15)
    ax.fill_between(dt, window, 0, where=(window < 0), color=theme.series[7], alpha=0.15)
    ax.text(
        40,
        float(jnp.max(window)) * 0.75,
        "pre before post\nstrengthens",
        fontsize=9,
        color=theme.secondary,
    )
    ax.text(
        -78,
        float(jnp.min(window)) * 0.75,
        "post before pre\nweakens",
        fontsize=9,
        color=theme.secondary,
        va="top",
    )
    # Typographic minus, not a hyphen: this is display text on an axis.
    ax.set_xlabel("t_post − t_pre  (ms)")  # noqa: RUF001
    ax.set_ylabel("Δw")
    ax.set_title("STDP window", loc="left", fontsize=11)

    # 2. Short-term plasticity: the same 20 Hz train through different synapses.
    ax = viz._prepare(axes[1], theme, None)
    train = jnp.zeros((420, 1)).at[jnp.arange(0, 420, 50), 0].set(1.0)
    for index, name in enumerate(("depressing", "facilitating", "F3_mixed")):
        out, _ = jp.unroll(jp.TsodyksMarkram.preset(name), train)
        amps = out[out[:, 0] > 0][:, 0]
        ax.plot(
            range(1, len(amps) + 1),
            amps,
            marker="o",
            markersize=6,
            linewidth=2.0,
            color=theme.series[index],
            label=name.replace("_", " "),
        )
    ax.set_xlabel("spike number in a 20 Hz train")
    ax.set_ylabel("transmitted amplitude")
    ax.set_title("Tsodyks-Markram short-term plasticity", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=theme.secondary)

    # 3. Dopamine-modulated STDP: reward arriving long after the spike pair.
    ax = viz._prepare(axes[2], theme, None)
    delays = [0, 100, 250, 500, 1000, 1500, 2000]
    steps = 2600
    rule = jp.DopamineSTDP()
    pre = jnp.zeros((steps, 1, 1)).at[100, 0, 0].set(1.0)
    post = jnp.zeros((steps, 1, 1)).at[105, 0, 0].set(1.0)
    changes = []
    for delay in delays:
        reward = jnp.zeros(steps).at[100 + delay].set(1.0)
        updated, _ = rule(jnp.array([[0.5]]), pre, post, reward)
        changes.append(float(updated[0, 0] - 0.5))
    ax.plot(
        delays,
        changes,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=theme.series[0],
        solid_capstyle="round",
    )
    ax.axhline(0, color=theme.grid, linewidth=1.2)
    ax.set_xlabel("reward delay after the spike pair (ms)")
    ax.set_ylabel("Δw")
    ax.set_title("Dopamine-modulated STDP — the distal reward problem", loc="left", fontsize=11)
    ax.text(
        0.99,
        0.9,
        "reward 2 s late still reinforces",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=theme.secondary,
    )

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sample, digit, real = load_sample()
    label = f"SHD digit {digit}" if real else "synthetic"
    net = build(sample.shape[1], jax.random.key(0))

    written = []
    for mode, theme in (("light", viz.Theme.light()), ("dark", viz.Theme.dark())):
        for name, figure in (
            ("gallery", gallery(theme, sample, net, label, real)),
            ("izhikevich", izhikevich_panel(theme)),
            ("plasticity", plasticity_panel(theme)),
        ):
            path = args.out / f"{name}_{mode}.png"
            figure.savefig(path, dpi=110, facecolor=theme.surface)
            plt.close(figure)
            written.append(path)

    for path in written:
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
