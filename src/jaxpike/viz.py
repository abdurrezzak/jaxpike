"""Visualization for spiking networks.

Needs matplotlib (``pip install "jaxpike[viz]"``). Every function takes an optional `ax` and
returns the `Axes` it drew on, so plots compose into larger figures instead of owning them.

The chart forms follow from what the data is doing rather than from habit. A raster is
identity-over-time, so it is a mark per event and nothing else. A membrane trace is
change-over-time, so it is a line with the threshold drawn as a reference rather than as a
second series. `layer_rates` is a magnitude comparison across a handful of categories, so it
is a bar chart — and it is the most useful function here, because it diagnoses the failure
mode that kills deep SNNs: activity decaying layer by layer until the network is silent and
has no gradient anywhere. Weight matrices are polarity data (excitatory versus inhibitory),
so they get a diverging blue-red map with a neutral midpoint, never a rainbow.

Colours come from a validated palette: sequential encodings use one hue light-to-dark,
diverging uses two poles with a gray middle, and text never wears a series colour.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# Sequential ramp: one hue, light to dark. For magnitude only.
_BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
_RED_RAMP = ["#7a1c1c", "#a52a2a", "#c93a39", "#e34948", "#ea6d6c", "#f19392", "#f8bab9"]


@dataclass(frozen=True)
class Theme:
    """Surface, ink and series colours for one mode.

    Dark is a selected set stepped for the dark surface, not an automatic inversion of light.
    """

    surface: str
    text: str
    secondary: str
    grid: str
    series: tuple[str, ...]
    midpoint: str

    @staticmethod
    def light() -> Theme:
        return Theme(
            surface="#fcfcfb",
            text="#0b0b0b",
            secondary="#52514e",
            grid="#e4e3e0",
            series=(
                "#2a78d6",
                "#eb6834",
                "#1baf7a",
                "#eda100",
                "#e87ba4",
                "#008300",
                "#4a3aa7",
                "#e34948",
            ),
            midpoint="#f0efec",
        )

    @staticmethod
    def dark() -> Theme:
        return Theme(
            surface="#1a1a19",
            text="#ffffff",
            secondary="#c3c2b7",
            grid="#383835",
            series=(
                "#3987e5",
                "#d95926",
                "#199e70",
                "#c98500",
                "#d55181",
                "#008300",
                "#9085e9",
                "#e66767",
            ),
            midpoint="#383835",
        )


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('plotting needs matplotlib: pip install "jaxpike[viz]"') from exc
    return plt


def _prepare(ax, theme: Theme, figsize):
    plt = _mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(theme.surface)
    if ax.figure is not None:
        ax.figure.set_facecolor(theme.surface)
    # Recessive frame: keep the two axes that carry meaning, drop the box.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.tick_params(colors=theme.secondary, labelsize=9, length=3)
    ax.xaxis.label.set_color(theme.secondary)
    ax.yaxis.label.set_color(theme.secondary)
    ax.title.set_color(theme.text)
    return ax


def _sequential_cmap(theme: Theme):
    from matplotlib.colors import LinearSegmentedColormap

    stops = _BLUE_RAMP if theme.surface == Theme.light().surface else list(reversed(_BLUE_RAMP))
    return LinearSegmentedColormap.from_list("jaxpike_seq", stops)


def _diverging_cmap(theme: Theme):
    """Blue (inhibitory) to gray to red (excitatory), equal steps per arm.

    Both arms must run dark-at-the-extreme to light-at-the-midpoint and carry the same number
    of steps, or the colour scale compresses one polarity relative to the other and a
    symmetric weight distribution reads as if it were skewed.
    """
    from matplotlib.colors import LinearSegmentedColormap

    blue = _BLUE_RAMP[::2][:7]  # 7 steps to match the red ramp
    stops = [*reversed(blue), theme.midpoint, *reversed(_RED_RAMP)]
    return LinearSegmentedColormap.from_list("jaxpike_div", stops)


def _as_2d(spikes, name: str) -> np.ndarray:
    """Accept `(T, N)` or `(T, B, N)`; a batched array is reduced to its first example."""
    array = np.asarray(spikes)
    if array.ndim == 3:
        array = array[:, 0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be (T, N) or (T, B, N), got shape {np.shape(spikes)}")
    return array


def raster(
    spikes,
    *,
    ax=None,
    theme: Theme | None = None,
    max_neurons: int = 200,
    figsize=(9, 4),
    title: str | None = "Spike raster",
):
    """The canonical SNN plot: one mark per spike, time across, neuron index up.

    A raster's job is identity and timing, not magnitude, so it gets marks and nothing else --
    no fill, no interpolation, no colour scale. Above `max_neurons` the rows are drawn as an
    image instead of scattered marks, because individual markers stop being resolvable and
    overplotting would misrepresent the density.
    """
    theme = theme or Theme.light()
    data = _as_2d(spikes, "spikes")
    ax = _prepare(ax, theme, figsize)
    steps, neurons = data.shape

    if neurons <= max_neurons:
        times, units = np.nonzero(data)
        ax.scatter(
            times,
            units,
            s=3,
            marker="|",
            linewidths=0.9,
            color=theme.series[0],
            rasterized=len(times) > 20000,
        )
    else:
        # Two colours, not a ramp. Spikes are binary, so a sequential scale would paint every
        # empty slot in the ramp's lightest step and "no spike" would read as "a little bit".
        from matplotlib.colors import ListedColormap

        ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=ListedColormap([theme.surface, theme.series[0]]),
            vmin=0,
            vmax=1,
            extent=(0, steps, 0, neurons),
        )

    ax.set_xlim(0, steps)
    ax.set_ylim(-0.5, neurons - 0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("neuron")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    density = float(np.mean(data != 0))
    ax.text(
        0.995,
        1.02,
        f"{density:.1%} active",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=theme.secondary,
    )
    return ax


def membrane(
    voltage,
    *,
    spikes=None,
    threshold: float | None = None,
    neuron: int = 0,
    ax=None,
    theme: Theme | None = None,
    figsize=(9, 3.2),
    title: str | None = "Membrane potential",
):
    """One neuron's membrane over time, with the threshold as a reference line.

    The threshold is drawn as an annotated rule rather than a second series: it is a constant
    the reader compares against, not data with its own identity, so it stays in muted ink.
    """
    theme = theme or Theme.light()
    trace = _as_2d(voltage, "voltage")[:, neuron]
    ax = _prepare(ax, theme, figsize)

    ax.plot(trace, linewidth=2.0, color=theme.series[0], solid_capstyle="round")
    if threshold is not None:
        ax.axhline(threshold, linewidth=1.2, linestyle="--", color=theme.secondary, alpha=0.8)
        ax.text(
            0.998,
            threshold,
            "threshold ",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color=theme.secondary,
        )
    if spikes is not None:
        times = np.nonzero(_as_2d(spikes, "spikes")[:, neuron])[0]
        if len(times):
            ax.scatter(
                times,
                np.full(len(times), np.max(trace)),
                s=28,
                marker="v",
                color=theme.series[1],
                zorder=3,
                label="spike",
            )
            # One direct label on the first marker, rather than a legend box: a legend
            # placed inside lands on the trace, and one placed outside detaches from the
            # panel entirely when the plot is a cell in a larger grid.
            ax.annotate(
                "spike",
                xy=(times[0], np.max(trace)),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
                color=theme.secondary,
            )

    ax.set_xlim(0, len(trace))
    ax.set_xlabel("timestep")
    ax.set_ylabel("potential")
    if title:
        ax.set_title(f"{title} — neuron {neuron}", loc="left", fontsize=11)
    return ax


def layer_rates(
    rates,
    *,
    labels=None,
    ax=None,
    theme: Theme | None = None,
    healthy=(0.02, 0.5),
    figsize=(7, 3.6),
    title: str | None = "Firing rate by layer",
):
    """Firing rate per layer — the diagnostic for a network that has gone silent.

    Deep SNNs fail by activity decaying multiplicatively with depth until the output layer
    never fires, at which point there is no gradient anywhere and training cannot recover.
    Reading the rates side by side makes that visible immediately: a healthy network holds
    roughly level, a dying one falls off a cliff.

    Bars outside `healthy` are drawn in the alert hue and labelled, so the problem is carried
    by position and text as well as colour.

    Pass rates from `jaxpike.density` per layer, or use `layer_rates_from(net, xs)`.
    """
    theme = theme or Theme.light()
    values = [float(r) for r in rates]
    labels = list(labels) if labels is not None else [f"layer {i}" for i in range(len(values))]
    ax = _prepare(ax, theme, figsize)

    low, high = healthy
    ok, alert = theme.series[0], theme.series[7]
    colors = [ok if low <= v <= high else alert for v in values]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.62)
    for bar in bars:
        bar.set_linewidth(2.0)
        bar.set_edgecolor(theme.surface)  # 2px surface gap between adjacent fills

    ax.axhspan(low, high, color=theme.series[0], alpha=0.07, zorder=0)
    # Labels stay horizontal. Rotating them to fit collides with neighbouring bars and
    # overflows the axes, and the warning is the part most worth reading.
    headroom = max(max(values, default=0), high) * 0.035
    for index, value in enumerate(values):
        ax.text(
            index,
            value + headroom,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=theme.secondary,
        )
        warning = "silent" if value < low else ("saturating" if value > high else None)
        if warning:
            ax.text(
                index,
                value + headroom * 5,
                warning,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=alert,
            )

    ax.set_xticks(range(len(values)), labels, fontsize=9)
    ax.set_ylabel("fraction of steps firing")
    ax.set_ylim(0, max(max(values, default=0) * 1.5, high * 1.3))
    ax.grid(axis="y", color=theme.grid, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return ax


def layer_rates_from(net, xs, *, runner=None, **kwargs):
    """Run `net` on `xs` and plot the firing rate after every spiking layer."""
    from .execution import unroll
    from .layers import Sequential
    from .neurons import ALIF, LIF, Izhikevich, LinearLIF

    runner = runner or unroll
    spiking = (LIF, ALIF, LinearLIF, Izhikevich)
    layers = net.layers if isinstance(net, Sequential) else (net,)

    rates, labels = [], []
    for index, layer in enumerate(layers):
        if not isinstance(layer, spiking):
            continue
        output, _ = runner(Sequential(*layers[: index + 1]), xs)
        rates.append(float(jnp.mean(output != 0)))
        labels.append(f"{type(layer).__name__}\n@{index}")
    if not rates:
        raise ValueError("network contains no spiking layers to measure")
    return layer_rates(rates, labels=labels, **kwargs)


def rate_heatmap(
    spikes,
    *,
    bins: int = 50,
    ax=None,
    theme: Theme | None = None,
    figsize=(9, 4),
    title: str | None = "Firing rate over time",
):
    """Neurons against binned time, coloured by firing rate.

    Magnitude, so one hue light-to-dark. Binning is explicit rather than implied by pixel
    resolution, so the colour of a cell always means the same thing.
    """
    theme = theme or Theme.light()
    data = _as_2d(spikes, "spikes")
    steps, neurons = data.shape
    bins = max(1, min(bins, steps))
    edges = np.linspace(0, steps, bins + 1).astype(int)
    binned = np.stack(
        [
            data[a:b].mean(axis=0) if b > a else np.zeros(neurons)
            for a, b in itertools.pairwise(edges)
        ]
    )

    ax = _prepare(ax, theme, figsize)
    image = ax.imshow(
        binned.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=_sequential_cmap(theme),
        extent=(0, steps, 0, neurons),
        vmin=0,
    )
    bar = ax.figure.colorbar(image, ax=ax, pad=0.015)
    bar.set_label("firing rate", color=theme.secondary, fontsize=9)
    bar.ax.tick_params(colors=theme.secondary, labelsize=9)
    bar.outline.set_visible(False)
    ax.set_xlabel("timestep")
    ax.set_ylabel("neuron")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return ax


def weights(
    matrix,
    *,
    ax=None,
    theme: Theme | None = None,
    figsize=(5.5, 4.5),
    title: str | None = "Weights",
):
    """Weight matrix as a diverging map, symmetric about zero.

    Weights are polarity data -- excitatory above zero, inhibitory below -- so the scale is
    two poles with a neutral midpoint and limits forced symmetric. An asymmetric or
    single-hue scale here would make zero land at an arbitrary colour and read as if the
    network were biased when it is not.
    """
    theme = theme or Theme.light()
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"weights must be 2-D, got shape {array.shape}")
    limit = float(np.max(np.abs(array))) or 1.0

    ax = _prepare(ax, theme, figsize)
    image = ax.imshow(
        array,
        aspect="auto",
        cmap=_diverging_cmap(theme),
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    bar = ax.figure.colorbar(image, ax=ax, pad=0.015)
    bar.set_label("weight", color=theme.secondary, fontsize=9)
    bar.ax.tick_params(colors=theme.secondary, labelsize=9)
    bar.outline.set_visible(False)
    ax.set_xlabel("input")
    ax.set_ylabel("output")
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    return ax


__all__ = [
    "Theme",
    "architecture",
    "layer_rates",
    "layer_rates_from",
    "membrane",
    "raster",
    "rate_heatmap",
    "weights",
]


def _topology(net, input_shape):
    """Normalize a Sequential or Graph into (nodes, edges, back-edges, output, shapes)."""
    from .graph import INPUT, Graph
    from .layers import Sequential

    if isinstance(net, Graph):
        labels = {name: type(layer).__name__ for name, layer in net.nodes.items()}
        shapes = net.shapes(input_shape) if input_shape else {}
        return labels, list(net.edges), set(net.back), net.output, shapes

    layers = net.layers if isinstance(net, Sequential) else (net,)
    labels = {str(i): type(layer).__name__ for i, layer in enumerate(layers)}
    edges = [(INPUT, "0")] + [(str(i), str(i + 1)) for i in range(len(layers) - 1)]
    shapes = {}
    if input_shape:
        shape = input_shape
        shapes[INPUT] = shape
        for i, layer in enumerate(layers):
            shape = layer.out_shape(shape)
            shapes[str(i)] = shape
    return labels, edges, set(), str(len(layers) - 1), shapes


def architecture(
    net,
    *,
    input_shape=None,
    ax=None,
    theme: Theme | None = None,
    figsize=None,
    title: str | None = "Architecture",
):
    """Draw the network's topology: which layer feeds which.

    Works for `Sequential` and `Graph`. Feedback edges -- the ones a `Graph` delays by a
    timestep to break a cycle -- are drawn curved, dashed and in the alert hue, because the
    difference between a feedforward and a recurrent network is the single most important
    thing a topology diagram can communicate, and it must not rest on the reader tracing
    arrows.

    Pass `input_shape` to annotate each node with the shape it produces.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from .graph import INPUT

    theme = theme or Theme.light()
    labels, edges, back, output, shapes = _topology(net, input_shape)

    # Longest path from the input places each node in a column; nodes at equal depth stack.
    forward = [(s, d) for s, d in edges if (s, d) not in back]
    depth = {INPUT: 0}
    for _ in range(len(labels) + 1):
        for src, dst in forward:
            if src in depth:
                depth[dst] = max(depth.get(dst, 0), depth[src] + 1)
    for name in labels:
        depth.setdefault(name, 0)

    columns: dict[int, list[str]] = {}
    for name in [INPUT, *labels]:
        columns.setdefault(depth[name], []).append(name)

    position = {}
    for column, members in columns.items():
        for row, name in enumerate(members):
            position[name] = (column * 2.6, -(row - (len(members) - 1) / 2) * 1.5)

    width = max(depth.values()) + 1
    height = max(len(m) for m in columns.values())
    ax = _prepare(ax, theme, figsize or (max(7, width * 2.2), max(3.2, height * 1.5)))
    ax.set_axis_off()

    for src, dst in edges:
        x0, y0 = position[src]
        x1, y1 = position[dst]
        delayed = (src, dst) in back
        # Curve anything that does not connect adjacent columns. A skip connection drawn
        # straight passes behind the nodes it skips and becomes invisible, which makes the
        # diagram understate the topology -- the worst failure a topology diagram can have.
        span = abs(depth[dst] - depth[src])
        if delayed:
            rad = 0.42
        elif span > 1:
            rad = -0.18 - 0.06 * span
        else:
            rad = 0.0
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.62, y0),
                (x1 - 0.62, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.8 if delayed else 1.4,
                linestyle="--" if delayed else "-",
                color=theme.series[7] if delayed else theme.secondary,
                alpha=0.95 if delayed else 0.6,
                zorder=1 if span <= 1 else 4,
            )
        )

    for name in [INPUT, *labels]:
        x, y = position[name]
        is_input, is_output = name == INPUT, name == output
        face = theme.surface if is_input else (theme.series[0] if is_output else theme.surface)
        edge = theme.secondary if is_input else theme.series[0]
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.6, y - 0.3),
                1.2,
                0.6,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.8,
                zorder=2,
            )
        )
        ink = theme.surface if is_output else theme.text
        ax.text(
            x,
            y + 0.06,
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=ink,
            zorder=3,
        )
        subtitle = "input" if is_input else labels[name]
        ax.text(
            x,
            y - 0.13,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
            color=theme.surface if is_output else theme.secondary,
            zorder=3,
        )
        if shapes.get(name):
            ax.text(
                x,
                y - 0.42,
                # Multiplication sign, not the letter x: this is a shape label.
                "×".join(str(d) for d in shapes[name][1:]) or "scalar",  # noqa: RUF001
                ha="center",
                va="top",
                fontsize=7.5,
                color=theme.secondary,
                zorder=3,
            )

    ax.set_xlim(-1.2, max(x for x, _ in position.values()) + 1.2)
    ys = [y for _, y in position.values()]
    ax.set_ylim(min(ys) - 1.0, max(ys) + 0.95)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=theme.text)
    # Whether the network is recurrent goes in its own annotation rather than appended to the
    # title, which reads as a duplicate whenever the caller's title already says it.
    ax.text(
        0.999,
        1.0,
        "dashed = feedback, delayed one timestep" if back else "feedforward",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=theme.series[7] if back else theme.secondary,
    )
    return ax
