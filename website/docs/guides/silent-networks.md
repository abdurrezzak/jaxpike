---
id: silent-networks
title: Why deep SNNs go silent
sidebar_position: 2
---

# Why deep SNNs go silent

If your spiking network trains to chance and the loss barely moves, check this first. It is the
most common real failure mode in the field, and it has a one-line fix.

## The failure

Deep spiking networks have a failure mode that ANNs do not. Each layer emits binary spikes at
some rate; if a layer's drive lands below threshold it fires less than the layer before it, so
activity decays multiplicatively with depth until nothing reaches the output. **A silent network
has no gradient anywhere.** It is not slow training — it is no training, and it never recovers
on its own.

Measured in this library, a three-layer spiking convnet (`Conv2d` 2→8→16, `Dense` to 10,
`LinearLIF` at `tau=20`, `threshold=0.2`) with plain LeCun initialization:

| | layer 1 | layer 2 | layer 3 |
|---|---:|---:|---:|
| plain LeCun init | 0.045 | **0.000** | **0.000** |
| `gain=jp.lif_gain(20.0)` | 0.380 | 0.327 | 0.200 |

Dead by layer two, and every downstream gradient is exactly zero.

## The cause

jaxpike neurons use the normalized convention

```
v[t] = alpha*v[t-1] + (1 - alpha)*x[t]
```

which is an exponential moving average. For white-noise input, an EMA attenuates the signal's
standard deviation by `sqrt((1 - alpha)/(1 + alpha))` — averaging over a window cancels most of
the fluctuation. At `tau=20` that is a factor of **6.3**.

So weights initialized for unit-variance *activations*, which is what LeCun and He
initialization give you, produce membrane potentials six times smaller than intended, sitting
far below threshold. Nothing about the initialization is wrong; it is solving a problem the
membrane then undoes.

## The fix

```python
gain = jp.lif_gain(tau=20.0)     # 6.33

net = jp.Sequential(
    jp.Conv2d(2, 32, 3, key=k1, gain=gain),
    jp.LinearLIF(tau=20.0, threshold=0.2),
    jp.Pool2d(2),
    jp.Conv2d(32, 64, 3, key=k2, gain=gain),
    jp.LinearLIF(tau=20.0, threshold=0.2),
)
```

`lif_gain` returns `sqrt((1 + alpha)/(1 - alpha))`, exactly the compensating factor, and both
`Dense` and `Conv2d` accept it as `gain`. Larger `tau` averages over a longer window and needs
more gain: 6.3 at `tau=20`, 20.0 at `tau=200`.

## Diagnosing it

`viz.layer_rates_from` runs the network, plots the firing rate after every spiking layer, and
labels any layer that has gone silent or saturated:

```python
from jaxpike import viz

viz.layer_rates_from(net, xs)
```

Read the plot as follows. Rates falling toward zero with depth is the failure above — add the
gain. A flat profile in a healthy band (roughly 0.05 to 0.4) is what you want. Rates near 1.0
mean the opposite problem: the layer fires every timestep, carries no temporal information, and
the surrogate gradient is flat there too.

Without matplotlib, `jp.density(spikes)` gives the same number per layer:

```python
for i in range(len(net.layers)):
    out, _ = jp.unroll(jp.Sequential(*net.layers[: i + 1]), xs)
    print(i, float(jp.density(out)))
```

## When the gain is not enough

**Saturation instead of silence.** If layers fire near 1.0, lower the gain or raise the
threshold, and add a rate penalty to the loss:

```python
loss = jp.cross_entropy(logits, labels) + 0.1 * jp.rate_penalty(hidden_spikes, target=0.05)
```

**Recurrent connections need their own, much smaller gain.** A recurrent weight's output is
summed into the same membrane on the next timestep, so a gain sized for feedforward drive makes
the loop self-amplifying and the network saturates within a few timesteps. The SHD recurrent
model uses `gain=0.2` on the recurrent weight against `6.33` on the feedforward ones.

**Classifying on spike counts.** If the output layer counts spikes, a class that never fires
produces no gradient and can never learn to fire. Use a `LeakyIntegrator` readout with
`jp.max_membrane_logits`, which stays differentiable from the first timestep.

**Izhikevich neurons ignore all of this.** They work in millivolts with a resting potential
around −65 mV and input currents of order 1–20, so LIF-tuned initialization does nothing useful.
Scale weights for that range instead.
