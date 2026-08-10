---
id: plasticity
title: Plasticity
sidebar_position: 7
---

# Plasticity

Local learning and short-term synaptic dynamics. Unlike everything else in this library, these
rules use no loss function, no gradients and no backward pass — a synapse changes strength from
the relative timing of the spikes at its two ends. That locality is why neuromorphic hardware
can implement them directly.

The trade is real: STDP has no notion of a task. It extracts correlation structure from input
statistics but will not train a classifier on its own. The usual recipes are STDP pretraining
followed by a supervised readout, or a three-factor rule where a reward signal gates the
update.

## STDP

Pair-based spike-timing-dependent plasticity with exponential eligibility traces.

```python
jp.STDP(*, tau_pre=20.0, tau_post=20.0, a_plus=0.01, a_minus=0.012,
        w_min=0.0, w_max=1.0, dt=1.0)
```

| Argument | Meaning |
|---|---|
| `tau_pre`, `tau_post` | Decay constants of the pre- and postsynaptic traces, in timesteps. |
| `a_plus` | Potentiation amplitude, applied when post follows pre. |
| `a_minus` | Depression amplitude, applied when pre follows post. |
| `w_min`, `w_max` | Weights are clipped to this range after each update. |
| `dt` | Timestep, in the same units as the time constants. |

The rule consumes whole spike trains rather than single steps: `pre_spikes` and `post_spikes`
are `(T, batch, n_pre)` and `(T, batch, n_post)`, and the update is applied over the sequence.

```python
import jax.numpy as jnp
import jaxpike as jp

rule = jp.STDP(tau_pre=20.0, tau_post=20.0)
weight = 0.5 * jnp.ones((50, 100))                       # (post, pre)

weight, state = rule(weight, pre_spikes, post_spikes, learning_rate=1.0)
```

Pass `state` back in to continue across chunks; omit it and the traces start at rest. A fresh
state can also be built explicitly with `rule.init_state(batch, n_pre, n_post)`.

`a_minus` exceeds `a_plus` by default. Depression slightly stronger than potentiation is the
standard choice: with symmetric amplitudes, uncorrelated input drives weights upward until they
saturate at `w_max`.

Traces are read from `t-1` deliberately. A pre and post spike within the same timestep are
simultaneous rather than causally ordered, and should not potentiate.

### stdp_window

```python
jp.stdp_window(delta_t, *, tau_pre=20.0, tau_post=20.0, a_plus=0.01, a_minus=0.012)
```

The learning window `Δw(Δt)`, for plotting or for checking parameters against a published
figure. `delta_t` is post-spike time minus pre-spike time, so positive values are causal.

```python
import jax.numpy as jnp

delta_t = jnp.linspace(-100.0, 100.0, 401)
dw = jp.stdp_window(delta_t, tau_pre=20.0, tau_post=20.0)
```

## TsodyksMarkram

Short-term plasticity: depression and facilitation acting on transmission over hundreds of
milliseconds, without changing the underlying weight.

```python
jp.TsodyksMarkram(U=0.6, tau_d=800.0, tau_f=0.0, *, dt=1.0)
```

| Argument | Meaning |
|---|---|
| `U` | Baseline release probability. |
| `tau_d` | Recovery constant of the depleted resource pool. |
| `tau_f` | Facilitation constant. `0.0` disables facilitation, giving pure depression. |

`TsodyksMarkram` follows the ordinary state contract, so it drops into a `Sequential` between a
neuron and the layer it drives:

```python
state = rule.init_state(input_shape)
state, transmitted = rule(state, spikes)
```

Presets from the Markram characterization are available in `jp.MARKRAM_PRESETS` as
`(U, tau_d, tau_f)` tuples:

```python
jp.TsodyksMarkram(*jp.MARKRAM_PRESETS["depressing"])
```

| Preset | Behaviour |
|---|---|
| `depressing` | Successive spikes transmit progressively less. |
| `facilitating` | Successive spikes transmit progressively more. |
| `F1_facilitating`, `F2_depressing`, `F3_mixed` | The three cortical classes as characterized. |

## DopamineSTDP

Reward-modulated STDP: a three-factor rule where the coincidence of pre and post spikes leaves
a slowly decaying eligibility trace, and weight change occurs only when dopamine arrives.

```python
jp.DopamineSTDP(*, tau_pre=20.0, tau_post=20.0, tau_c=1000.0, tau_dopamine=200.0,
                a_plus=0.01, a_minus=0.0105, dopamine_per_reward=0.005,
                w_min=0.0, w_max=1.0, dt=1.0)
```

| Argument | Meaning |
|---|---|
| `tau_c` | Decay of the eligibility trace — how long a spike pair remains creditable. |
| `tau_dopamine` | Decay of the dopamine signal itself. |
| `dopamine_per_reward` | Dopamine released per unit of reward. |

```python
rule = jp.DopamineSTDP(tau_c=1000.0)

# reward is (T,): the signal delivered at each timestep.
weight, state = rule(weight, pre_spikes, post_spikes, reward)
```

`tau_c` is what solves the distal reward problem: a reward arriving a second after the spike
pair that earned it still finds the eligibility trace alive, so credit reaches the correct
synapse.

![Plasticity](/img/figures/plasticity_light.png)

## State objects

Each rule carries its traces in an explicit state object, returned as the second element of a
call and accepted back to continue across chunks. They are ordinary pytrees.

| type | carried by | holds |
|---|---|---|
| `STDPState` | `STDP` | pre- and postsynaptic eligibility traces |
| `DopamineState` | `DopamineSTDP` | the STDP traces, the slow eligibility trace, and dopamine |
| `MarkramState` | `TsodyksMarkram` | the resource pool and the release probability |

```python
state = rule.init_state(batch, n_pre, n_post)     # STDP and DopamineSTDP
state = rule.init_state(input_shape)              # TsodyksMarkram
```

## See also

- [Plasticity guide](../guides/plasticity.md) — worked examples and when to reach for each rule
- [Online learning](../guides/online-learning.md) — e-prop, for task-driven local learning
