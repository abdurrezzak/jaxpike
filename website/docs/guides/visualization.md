---
id: visualization
title: Visualization
sidebar_position: 8
---

# Visualization

`jaxpike.viz` needs matplotlib: `pip install "jaxpike[viz]"`.

<img src="/img/figures/gallery_light.png" alt="Visualization gallery" className="figure-light" />
<img src="/img/figures/gallery_dark.png" alt="Visualization gallery" className="figure-dark" />

*Real Spiking Heidelberg Digits data. Regenerate every figure with
`PYTHONPATH=. python examples/gallery.py`; each is rendered for light and dark mode.*

```python
from jaxpike import viz

viz.raster(spikes)                    # the canonical SNN plot
viz.membrane(voltage, spikes=spikes, threshold=1.0)
viz.layer_rates_from(net, xs)         # is my network silent?
viz.rate_heatmap(spikes)
viz.weights(net.layers[0].weight)
viz.architecture(net, input_shape=(1, 700))
```

Every function takes an optional `ax` and returns the `Axes` it drew on, so plots compose into
larger figures instead of owning them:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(9, 7))
viz.raster(spikes, ax=axes[0])
viz.membrane(voltage, threshold=1.0, ax=axes[1])
```

## Start with `layer_rates_from`

This is the one to reach for first, because it diagnoses the failure mode that kills deep SNNs.
It runs the network, plots the firing rate after every spiking layer, and labels any layer that
has gone silent or saturated.

```python
viz.layer_rates_from(net, xs)                            # runs with jp.unroll
viz.layer_rates_from(net, xs, runner=jp.unroll_parallel) # or your runner of choice
viz.layer_rates(rates, labels=["conv1", "conv2", "fc"])  # if you already have the numbers
```

The healthy band defaults to `(0.02, 0.5)`; rates below it mean the network is going silent and
has no gradient, rates above mean the layer fires nearly every timestep and carries no temporal
information. See [Why deep SNNs go silent](./silent-networks.md) for the fix.

## The other plots

**`raster(spikes)`** — one mark per spike, time across, neuron index up. A raster's job is
identity and timing, not magnitude, so it gets marks and nothing else: no fill, no
interpolation, no colour scale. Above `max_neurons=200` the rows are drawn as an image instead,
because individual markers stop being resolvable and overplotting would misrepresent density.

**`membrane(voltage, spikes=..., threshold=...)`** — one neuron's membrane over time, with the
threshold drawn as an annotated reference rule rather than a second series, and spikes marked
where they occurred. Pick the neuron with `neuron=`.

**`rate_heatmap(spikes, bins=50)`** — neurons against binned time, coloured by rate. Use it when
a raster is too dense to read.

**`weights(matrix)`** — a weight matrix as a diverging map, symmetric about zero. Weights are
polarity data (excitatory versus inhibitory), so they get a blue–red map with a neutral
midpoint, never a rainbow.

**`architecture(net, input_shape=...)`** — the topology, for `Sequential` and `Graph`. Feedback
edges are drawn distinctly, which is the fastest way to confirm the wiring you wrote is the
wiring you meant.

<img src="/img/figures/architecture_light.png" alt="Architecture diagrams" className="figure-light" />
<img src="/img/figures/architecture_dark.png" alt="Architecture diagrams" className="figure-dark" />

## Theming

```python
viz.raster(spikes, theme=viz.Theme.dark())
```

Dark is a selected set of colour steps for a dark surface, not an automatic inversion of the
light palette. Sequential encodings use one hue light-to-dark, diverging uses two poles with a
gray middle, and text never wears a series colour.

## Neuron dynamics

The seven Izhikevich firing patterns under identical drive, which is what `examples/gallery.py`
renders to sanity-check the presets:

<img src="/img/figures/izhikevich_light.png" alt="Izhikevich firing patterns" className="figure-light" />
<img src="/img/figures/izhikevich_dark.png" alt="Izhikevich firing patterns" className="figure-dark" />
