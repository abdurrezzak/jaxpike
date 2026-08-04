---
id: convnets
title: Spiking convnets
sidebar_position: 5
---

# Spiking convnets

```python
gain = jp.lif_gain(tau=20.0)      # not optional at this depth

net = jp.Sequential(
    jp.Conv2d(2, 32, 3, key=k1, gain=gain),      # 2 channels: DVS on/off events
    jp.LinearLIF(tau=20.0, threshold=0.2),
    jp.Pool2d(2),
    jp.Conv2d(32, 64, 3, key=k2, gain=gain),
    jp.LinearLIF(tau=20.0, threshold=0.2),
    jp.Pool2d(2),
    jp.Flatten(),
    jp.Dense(64 * 8 * 8, 10, key=k3, gain=gain),
    jp.LinearLIF(tau=20.0, threshold=0.2),
)
```

## Layout is NHWC

Inputs are `(time, batch, height, width, channels)`. PyTorch users will expect NCHW, but XLA's
convolutions are written for channels-last and NCHW forces a layout transpose around every op
on GPU and TPU. Porting a model means rewriting the layer construction anyway, so jaxpike takes
the faster layout rather than paying a permanent tax for familiarity.

If you are porting conv weights from a PyTorch model, transpose them. The kernel layout is
`(kh, kw, in_channels, out_channels)`.

## Everything here is stateless

`Conv2d`, `Pool2d` and `Flatten` hold no state and are applied independently at each timestep,
which has one large consequence: they parallelize over time for free. Internally they fold the
time axis into the batch, run one large op, and unfold. Since the layer holds no state, the
per-timestep and folded paths are guaranteed to agree.

That is what lets a spiking convnet run through `unroll_parallel` end to end — the only layers
carrying state are the neurons, and `LinearLIF` is reset-free.

```python
spikes, _ = jp.unroll_parallel(net, xs)     # works for the network above
```

Swap `LinearLIF` for `LIF` and the parallel path is gone; see
[Execution](./execution.md).

## Arguments

```python
jp.Conv2d(in_channels, out_channels, kernel_size=3, *, key,
          stride=1, padding="SAME", use_bias=True, gain=1.0)

jp.Pool2d(window=2, *, stride=None, mode="avg")    # stride defaults to window
jp.Flatten()
```

`kernel_size`, `stride` and `window` all accept either an int or an `(h, w)` tuple. Padding is
`"SAME"` or `"VALID"`.

**`mode="avg"` is the default pooling, and it is the better default for spikes.** Average
pooling over a binary map gives a graded value that carries gradient everywhere. Max pooling
routes gradient to a single winner, and on a mostly-zero spike map the winner is often a tie.
Max pooling also has no NIR equivalent, so a model using it cannot be exported.

## Sizing the Dense layer

`Flatten` produces `channels_last` ordering, so a `(h, w, c)` feature map becomes `h*w*c`
features in that order. Two `Pool2d(2)` layers on a 32×32 input give 8×8, hence
`64 * 8 * 8` above. `net.out_shape(input_shape)` computes it for you:

```python
jp.Sequential(*net.layers[:7]).out_shape((1, 32, 32, 2))
```

This ordering matters when exporting: NIR is channels-first, so a round trip has to permute the
following Dense layer's columns as well as transposing the conv weights. `jaxpike.nir` handles
it and there is a test dedicated to it. See [NIR](./nir.md).

## Initialization

Depth is where spiking convnets die. With plain LeCun init the network above fires 0.045, then
0.000, then 0.000 — no gradient anywhere. Pass `gain=jp.lif_gain(tau)` to every `Conv2d` and
`Dense`, and read [Why deep SNNs go silent](./silent-networks.md) before scaling depth further.

## Datasets

`jaxpike.data` currently covers SHD and SSC, which are 1-D channel streams rather than spatial
events. N-MNIST and DVS Gesture need a different loader shape and are not in yet; bin your own
events to `(samples, timesteps, height, width, channels)` as `uint8` on the host, and feed them
through `jp.iterate_batches`.
