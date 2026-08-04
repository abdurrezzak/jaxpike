---
id: layers
title: Layers and containers
sidebar_position: 3
---

# Layers and containers

## `Dense`

```python
jp.Dense(in_features, out_features, *, key, use_bias=True, gain=1.0)
```

Weight is `[out, in]`, matching the convention `jp.STDP` uses. Initialization is LeCun normal
(variance `1/fan_in`), which keeps pre-activation scale stable at init — and for a spiking net
that is what sets the initial firing rate.

`gain` multiplies the initialization. In a spiking network it is not decoration: pass
`gain=jp.lif_gain(tau)` or deep networks go silent. See
[Why deep SNNs go silent](../guides/silent-networks.md).

Stateless, so it runs per timestep and folds into the parallel path for free.

## `Sequential`

```python
jp.Sequential(*layers)
```

A straight chain. Shapes propagate through `out_shape`, so only the first layer needs to know
the input size. `net.layers` is a tuple, and slicing it gives you a prefix network — the usual
way to inspect intermediate activity:

```python
hidden, _ = jp.unroll(jp.Sequential(*net.layers[:2]), xs)
```

For anything that is not a chain — recurrence, skips, fan-in — use
[`Graph`](../guides/topologies.md).

## `Graph`

```python
jp.Graph(nodes: dict[str, layer], edges: list[tuple[str, str]], output: str)
```

Arbitrary topology. `"input"` is reserved for the external input. Multiple incoming edges sum;
an edge closing a cycle reads the previous timestep. Full treatment in
[Arbitrary topologies](../guides/topologies.md).

Useful attributes: `is_recurrent`, `back` (the delayed edges), `order` (evaluation order),
`incoming(name)`, `shapes(input_shape)`.

State is `GraphState(nodes, feedback)`.

## `Conv2d`

```python
jp.Conv2d(in_channels, out_channels, kernel_size=3, *, key,
          stride=1, padding="SAME", use_bias=True, gain=1.0)
```

NHWC layout — `(time, batch, height, width, channels)`. Kernel layout is
`(kh, kw, in, out)`. `kernel_size` and `stride` take an int or an `(h, w)` tuple; `padding` is
`"SAME"` or `"VALID"`.

## `Pool2d`

```python
jp.Pool2d(window=2, *, stride=None, mode="avg")
```

`stride` defaults to `window`. `mode="avg"` is the default and the better choice for binary
spikes: it gives a graded value that carries gradient everywhere, where max routes gradient to a
single winner that is often a tie on a mostly-zero map. Max pooling also cannot be exported to
NIR.

## `Flatten`

```python
jp.Flatten()
```

Flattens everything after the batch axis, channels-last. Relevant when exporting: NIR is
channels-first, so a round trip permutes the following `Dense` layer's columns.

## Stateless layers and time

`Conv2d`, `Pool2d` and `Flatten` hold no state and are applied independently at each timestep.
They implement `parallel_apply` by folding the time axis into the batch, running one large op,
and unfolding — so a spiking convnet built from these plus `LinearLIF` runs through
`unroll_parallel` end to end. Since the layer holds no state, the per-timestep and folded paths
are guaranteed to agree.

## Initialization

```python
jp.lif_gain(tau, dt=1.0) -> float
```

Returns `sqrt((1 + alpha)/(1 - alpha))`, the weight multiplier that restores unit membrane
variance for a LIF with this `tau`. 6.33 at `tau=20`, 20.0 at `tau=200`. Raises if `tau <= dt`.

Pass it as `gain=` to `Dense` and `Conv2d`. Recurrent weights are the exception — they need a
much smaller gain, because their output is summed into the same membrane on the next timestep.
