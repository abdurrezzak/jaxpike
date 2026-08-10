---
id: stdp-learning
title: Learning without gradients
sidebar_position: 3
---

# Learning without gradients

Everything else in this library trains by backpropagation through time. STDP does not. A
synapse changes strength from the relative timing of the spikes at its two ends — no loss
function, no gradients, no backward pass. That locality is why neuromorphic chips implement it
in hardware, and it is why a network can keep learning after deployment.

This tutorial shows STDP discovering structure nobody labelled, then extends it to learn from
delayed reward.

## The rule

Pre shortly before post strengthens a synapse; the reverse weakens it. Each side keeps an
exponentially decaying trace of its own spikes, and the update reads the other side's trace at
the moment a spike arrives.

```python
import jax
import jax.numpy as jnp
import jaxpike as jp

rule = jp.STDP(tau_pre=20.0, tau_post=20.0, a_plus=0.01, a_minus=0.012)
```

`a_minus` is larger than `a_plus` on purpose. With symmetric amplitudes, uncorrelated input
drives every weight upward until it saturates; slightly stronger depression is what keeps the
weight distribution from collapsing to the ceiling.

You can see the rule before running it. `stdp_window` returns Δw as a function of the spike
timing difference, which is the figure every STDP paper opens with:

```python
delta_t = jnp.linspace(-100.0, 100.0, 401)   # post time minus pre time
dw = jp.stdp_window(delta_t)                 # positive Δt potentiates
```

## Making it learn something

Give it input where a subset of channels fire together, buried in noise. Nothing labels that
subset; the correlation is the only signal.

```python
CHANNELS, N_POST, T, BATCH = 100, 1, 2000, 16
CORRELATED = 20          # channels 0..19 fire as a group


def make_input(key):
    k_group, k_noise = jax.random.split(key)
    group = (jax.random.uniform(k_group, (T, BATCH, 1)) < 0.05).astype(jnp.float32)
    noise = (jax.random.uniform(k_noise, (T, BATCH, CHANNELS)) < 0.05).astype(jnp.float32)
    correlated = jnp.broadcast_to(group, (T, BATCH, CORRELATED))
    return jnp.concatenate([correlated, noise[:, :, CORRELATED:]], axis=-1)
```

The postsynaptic neuron is driven by the input through the weights being learned, so
potentiation is self-reinforcing: channels that happen to drive it get stronger, which makes
them drive it more.

```python
weight = 0.5 * jnp.ones((N_POST, CHANNELS))
neuron = jp.LIF(tau=20.0, threshold=1.0)

key = jax.random.key(0)
for step in range(20):
    key, batch_key = jax.random.split(key)
    pre = make_input(batch_key)

    drive = pre @ weight.T                       # (T, batch, N_POST)
    post, _ = jp.unroll(neuron, drive)

    weight, _ = rule(weight, pre, post, learning_rate=0.05)

correlated = float(weight[0, :CORRELATED].mean())
background = float(weight[0, CORRELATED:].mean())
print(f"correlated {correlated:.3f}   background {background:.3f}")
# correlated 0.890   background 0.000
```

The correlated group ends near the ceiling and the background at the floor. No labels, no loss,
no gradient — the rule found the structure because those channels reliably preceded the
neuron's own spikes, and the rest did not.

**The learning rate is not a detail here.** STDP has no loss to overshoot, but it does have
weight bounds, and `a_minus` exceeds `a_plus`. Push too hard and *every* weight reaches `w_min`
before the correlated group can pull away — at `learning_rate=0.2` on this problem both groups
collapse to 0.000 and the run looks broken rather than merely mistuned. If your weights all
saturate at one bound, lower the rate before changing anything else.

## What STDP will not do

It has no notion of a task. It extracts correlation structure from input statistics, and if the
thing you care about is not the dominant correlation in your data, STDP will not find it. The
two standard remedies are to use STDP as unsupervised pretraining under a supervised readout,
or to gate the update with a reward signal — which is the next section.

## Learning from delayed reward

The credit assignment problem in its rawest form: a spike pair happens now, the reward arrives a
second later, and by then the traces that identified the responsible synapse have decayed.

`DopamineSTDP` solves it with a third factor. The coincidence writes a slowly decaying
*eligibility trace* rather than changing the weight, and the weight only moves when dopamine
arrives. Set `tau_c` to how long credit should remain assignable:

```python
rule = jp.DopamineSTDP(tau_c=1000.0, tau_dopamine=200.0)

reward = jnp.zeros((T,)).at[1500].set(1.0)      # arrives long after the spikes that earned it
weight, state = rule(weight, pre, post, reward)
```

With `tau_c = 1000` timesteps, a reward 500 steps late still finds the trace alive at roughly
`exp(-0.5)` of its original height, so the correct synapse is still identifiable. Setting
`tau_c` too short is the usual reason a reward-modulated rule fails to learn.

## Synapses that fatigue

Short-term plasticity is a different mechanism again: transmission strength changes over
hundreds of milliseconds without the underlying weight changing at all. A depressing synapse
transmits less with each spike in a burst; a facilitating one transmits more.

```python
rule = jp.TsodyksMarkram(*jp.MARKRAM_PRESETS["depressing"])

state = rule.init_state((BATCH, CHANNELS))
state, transmitted = rule(state, pre[0])
```

The presets come from Markram's characterization of cortical synapses, so
`F1_facilitating`, `F2_depressing` and `F3_mixed` correspond to measured classes rather than to
invented parameters.

`TsodyksMarkram` follows the ordinary state contract, so it drops straight into a `Sequential`
between a neuron and the layer it drives, and composes with gradient training normally.

## See also

- [Plasticity reference](../reference/plasticity.md) — every argument, and the state objects
- [Online learning](../guides/online-learning.md) — e-prop, for local learning *with* a task
