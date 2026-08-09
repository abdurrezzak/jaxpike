---
id: first-network
title: Your first spiking network
sidebar_position: 1
---

# Your first spiking network

This tutorial builds and trains a spiking classifier from nothing, explaining each piece of the
machinery as it appears. It assumes JAX but no spiking background.

By the end you will have trained a network on a synthetic temporal task and will understand
why each of the four ingredients — time, state, thresholds and surrogate gradients — is there.

## What makes a spiking network different

An ordinary network maps an input to an output in one shot. A spiking network runs over
**time**: at each timestep a neuron integrates its input into a membrane potential, and when
that potential crosses a threshold the neuron emits a spike and resets.

Two consequences follow, and they are the whole of what you need to learn:

1. **Neurons carry state.** The membrane potential persists between timesteps, so a spiking
   network is a recurrent network even when its wiring is feedforward.
2. **The activation is a step function.** Its derivative is zero everywhere and undefined at
   the threshold, so gradient descent cannot work on it directly. A *surrogate gradient*
   substitutes a smooth function in the backward pass.

## A task with temporal structure

A spiking network is only worth using when the answer depends on *when* things happen. So
rather than a static task, classify which of two input channels fired first — impossible to
solve without temporal information.

```python
import jax
import jax.numpy as jnp

TIMESTEPS, CHANNELS = 50, 20


def make_batch(key, batch=64):
    """Channel group 0 fires early and group 1 late, or the reverse."""
    k_label, k_noise = jax.random.split(key)
    labels = jax.random.bernoulli(k_label, 0.5, (batch,)).astype(jnp.int32)

    time = jnp.arange(TIMESTEPS)[:, None, None]
    early = (time < TIMESTEPS // 2)
    group = jnp.arange(CHANNELS)[None, None, :] < CHANNELS // 2

    # Label 0: first group early. Label 1: second group early.
    active = jnp.where(labels[None, :, None] == 0, group == early, group != early)
    noise = jax.random.uniform(k_noise, (TIMESTEPS, batch, CHANNELS))
    return (noise < jnp.where(active, 0.3, 0.02)).astype(jnp.float32), labels
```

Inputs are `(time, batch, channels)`. Time leads, which is the convention throughout the
library and what lets the execution functions scan over the leading axis.

## The model

```python
import jaxpike as jp

k1, k2 = jax.random.split(jax.random.key(0))

net = jp.Sequential(
    jp.Dense(CHANNELS, 64, key=k1),
    jp.LIF(tau=20.0),
    jp.Dense(64, 2, key=k2),
    jp.LeakyIntegrator(tau=20.0),
)
```

Four layers, two of which are stateful.

**`Dense`** is an ordinary linear map, applied identically at every timestep.

**`LIF`** is the leaky integrate-and-fire neuron. Its membrane follows
`v[t] = α·v[t-1] + (1-α)·x[t]` with `α = exp(-1/tau)`, emits a spike when `v` crosses the
threshold, and subtracts the threshold from the membrane afterwards. `tau=20.0` means the
membrane forgets its input over roughly 20 timesteps — long enough to bridge the temporal gap
in this task.

**`LeakyIntegrator`** is a LIF without the spike: it integrates and leaks but never fires. It is
the standard readout, because a classifier wants a graded score rather than a binary one.

## Running it

```python
xs, labels = make_batch(jax.random.key(1))
membrane, final_state = jp.unroll(net, xs)

membrane.shape     # (50, 64, 2) — the readout at every timestep
```

`unroll` scans the network over the leading axis and returns every timestep's output plus the
final state. State is explicit and functional: nothing is stored on the module, which is what
makes the network safe to `jit`, `vmap` and differentiate.

To turn `(time, batch, 2)` into logits, reduce over time. Summing the readout membrane weights
the whole trajectory:

```python
logits = jnp.sum(membrane, axis=0)
```

## Training

The loss is ordinary cross-entropy. The only spiking-specific part is invisible: the threshold
comparison inside `LIF` has a surrogate derivative attached, so gradients flow through it.

```python
import equinox as eqx
import optax

optimizer = optax.adam(1e-3)
params, static = eqx.partition(net, eqx.is_inexact_array)
opt_state = optimizer.init(params)


def loss_fn(params, xs, labels):
    membrane, _ = jp.unroll(eqx.combine(params, static), xs)
    logits = jnp.sum(membrane, axis=0)
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


@eqx.filter_jit
def train_step(params, opt_state, xs, labels):
    loss, grads = jax.value_and_grad(loss_fn)(params, xs, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return eqx.apply_updates(params, updates), opt_state, loss


key = jax.random.key(2)
for step in range(500):
    key, batch_key = jax.random.split(key)
    xs, labels = make_batch(batch_key)
    params, opt_state, loss = train_step(params, opt_state, xs, labels)
    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")

net = eqx.combine(params, static)
```

The loss should fall from about 0.69 — chance for two classes — toward zero.

## Checking the network is alive

The most common failure in spiking networks is silence: if no neuron fires, no gradient exists
anywhere and training cannot recover. Check the firing rates before debugging anything else:

```python
from jaxpike import viz

viz.layer_rates_from(net, xs)
```

Healthy rates sit somewhere around 5–30%. A layer at 0.000 is dead and a layer near 1.0 is
saturated, and both are fatal. If a layer is silent, see
[why deep SNNs go silent](../guides/silent-networks.md) — the fix is usually one argument.

## Choosing a surrogate gradient

`LIF` defaults to `FastSigmoid`. The choice matters less than beginners expect, but it is worth
knowing what the alternatives do:

```python
jp.LIF(tau=20.0, surrogate=jp.ATan())          # wide support, well-conditioned
jp.LIF(tau=20.0, surrogate=jp.Triangular())    # compact support, cheapest backward
jp.LIF(tau=20.0, surrogate=jp.Boxcar())        # the straight-through estimator
```

A surrogate with wide support passes gradient to neurons far from threshold, which helps a
silent network recover but blurs credit assignment. Compact support does the reverse.

## Where to go next

- [Train SHD end to end](../getting-started/training-shd.md) — a real dataset and a real result
- [Execution strategies](../guides/execution.md) — memory and speed on longer sequences
- [Topologies](../guides/topologies.md) — recurrence and skip connections with `Graph`
