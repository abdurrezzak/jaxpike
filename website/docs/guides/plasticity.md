---
id: plasticity
title: Plasticity — STDP and dynamic synapses
sidebar_position: 7
---

# Plasticity

Three mechanisms live in `jaxpike.plasticity`, and they are different kinds of thing:
`STDP` is an unsupervised learning rule, `DopamineSTDP` is a three-factor reward-modulated
rule, and `TsodyksMarkram` is not a learning rule at all — it is a synapse whose efficacy
changes within a stimulus.

<img src="/img/figures/plasticity_light.png" alt="Plasticity mechanisms" className="figure-light" />
<img src="/img/figures/plasticity_dark.png" alt="Plasticity mechanisms" className="figure-dark" />

## STDP is a different animal from the rest of the library

Everything else here trains by backpropagation through time with surrogate gradients: a global
loss, gradients pushed backwards through the whole unrolled network. STDP is **local and
unsupervised.** A synapse changes strength based only on the relative timing of the spikes at
its two ends — pre before post strengthens, post before pre weakens. No loss function, no
gradients, no backward pass, no information from anywhere else in the network.

That locality is why neuromorphic chips can implement it in hardware and why it is the standard
model of biological learning. The trade is real: **STDP has no notion of a task.** It extracts
correlation structure from input statistics, which is useful for unsupervised feature learning
and for studying a biologically plausible mechanism, but on its own it will not train a
classifier the way BPTT will. The usual recipes are STDP pretraining with a supervised readout
on top, or a three-factor rule where reward gates the update.

## `STDP`

```python
stdp = jp.STDP(tau_pre=20.0, tau_post=20.0, a_plus=0.01, a_minus=0.012,
               w_min=0.0, w_max=1.0)

new_weight, traces = stdp(weight, pre_spikes, post_spikes, learning_rate=1.0)
```

`weight` is `[post, pre]`, matching `Dense`. `pre_spikes` and `post_spikes` are
`[T, batch, units]` and must cover the same timesteps. Returns the updated weight and the final
traces, so a long sequence can be continued by passing `state=traces` back in. `stdp.step` is
the single-timestep form if you want to interleave it with your own loop.

The rule is the standard pair-based one with exponential eligibility traces:

```
trace_pre[t]  = exp(-dt/tau_pre)  * trace_pre[t-1]  + pre_spikes[t]
trace_post[t] = exp(-dt/tau_post) * trace_post[t-1] + post_spikes[t]

dw += a_plus  * outer(post_spikes[t], trace_pre[t-1])     # post after pre -> strengthen
dw -= a_minus * outer(trace_post[t-1], pre_spikes[t])     # pre after post -> weaken
```

Traces from `t-1` are used deliberately: a pre and post spike in the same timestep are
simultaneous, not causally ordered, and should not potentiate.

**`a_minus` larger than `a_plus` is the usual choice, and it matters.** Symmetric rates tend to
run away, because a strengthened synapse makes its postsynaptic neuron fire more, which
strengthens it further. A slight depression bias plus the `w_min`/`w_max` clamp keeps it in
check.

Plot the window before running anything, to check a configuration does what you think:

```python
import jax.numpy as jnp
delta_t = jnp.linspace(-100, 100, 400)     # t_post - t_pre
dw = jp.stdp_window(delta_t, tau_pre=20.0, tau_post=20.0)
```

## `DopamineSTDP` — learning from delayed reward

Plain STDP changes a weight the moment two spikes coincide, which cannot explain learning from
*delayed* reward: by the time reward arrives seconds later, the responsible spike pair is long
gone and a million irrelevant ones have happened since.

Izhikevich's (2007) answer is to insert a slow variable between them:

```
eligibility c:  STDP writes into c, not into the weight.  c decays with tau_c (~1 s)
dopamine d:     reward raises d.                          d decays with tau_dopamine (~200 ms)
weight:         dw/dt = c * d
```

The weight moves only where eligibility and dopamine **overlap**. A spike pair leaves a tag
that persists for about a second; if reward arrives inside that window the synapse is
reinforced, and if not the tag simply decays. Random firing during the wait does not wash the
tag out, because STDP's own window is only about 20 ms wide, so uncorrelated spikes contribute
near-zero net eligibility.

```python
rule = jp.DopamineSTDP(tau_c=1000.0, tau_dopamine=200.0, dopamine_per_reward=5e-3)
new_weight, state = rule(weight, pre_spikes, post_spikes, reward)   # reward is [T]
```

Defaults are the paper's, in milliseconds, so pair them with `dt=1.0` for a 1 ms step.

## `TsodyksMarkram` — synapses that fatigue and facilitate

Not a learning rule: nothing here is remembered across a stimulus. It is a *dynamic synapse*
whose efficacy changes over tens to hundreds of milliseconds because vesicles deplete and
presynaptic calcium accumulates, then relaxes back. The consequence is that a synapse transmits
a spike *train* differently depending on its rate and history — the point of the Markram et al.
result that the same axon signals differently to different targets.

Two variables per presynaptic neuron:

- `u` — utilization, the fraction of available resources released per spike. Decays to 0 with
  `tau_f`; each spike pushes it up by `U*(1-u)`. This is **facilitation**.
- `x` — resources available. Recovers toward 1 with `tau_d`; each spike consumes `u*x`. This is
  **depression**.

The transmitted amplitude is `u*x` rather than a binary 1, so the output is graded. Place it
directly after a spiking layer and before the weights it modulates:

```python
net = jp.Sequential(
    jp.Dense(128, 128, key=k1),
    jp.LIF(tau=20.0),
    jp.TsodyksMarkram.preset("depressing"),
    jp.Dense(128, 10, key=k2),
    jp.LeakyIntegrator(tau=20.0),
)
```

Presets, from Tsodyks–Pawelzik–Markram (1998) and Markram, Wang & Tsodyks (1998), with time
constants in milliseconds:

| Preset | U | tau_d | tau_f |
|---|---:|---:|---:|
| `depressing` | 0.60 | 800 | 0 |
| `facilitating` | 0.03 | 130 | 530 |
| `F1_facilitating` | 0.16 | 45 | 376 |
| `F2_depressing` | 0.25 | 706 | 21 |
| `F3_mixed` | 0.32 | 144 | 62 |

With `tau_f = 0` facilitation is off and the synapse purely depresses.

Both variables are affine in the spike train, so `TsodyksMarkram` implements `parallel_apply`
and does not block [parallel-in-time execution](./execution.md).

## Export

None of these mechanisms have a NIR equivalent, so a model using short-term plasticity or an
adaptive threshold cannot be exported. `jaxpike.nir` raises rather than exporting something
different. See [NIR](./nir.md).
