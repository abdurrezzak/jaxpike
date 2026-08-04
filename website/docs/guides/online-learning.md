---
id: online-learning
title: Online learning with e-prop
sidebar_position: 6
---

# Online learning with e-prop

BPTT stores every timestep and then walks backwards through them. That costs `O(T)` memory,
cannot produce an update until the sequence ends, and has no implementation on neuromorphic
hardware, which sees a spike stream and cannot run time in reverse.

**e-prop** (Bellec et al., 2020) replaces the backward-in-time pass with a local eligibility
trace carried forward with the network, so memory is `O(1)` in `T`.

## Using it

```python
def step_loss(output_t, labels):          # ONE timestep, shaped (batch, units)
    return jp.cross_entropy(output_t, labels)

value_and_grad = jp.eprop_value_and_grad(step_loss)
loss, grads = value_and_grad(model, xs, labels)
```

`eprop_value_and_grad` is a drop-in alternative to `equinox.filter_value_and_grad`, so a
training loop switches learning rule by changing one line. The lower-level form is
`jp.eprop_grads(net, xs, step_loss) -> (grads, loss)`.

## The per-timestep loss is not a style choice

`step_loss` must take a single timestep. That signature *is* the memory guarantee: the weight
gradient is accumulated inside the scan, so nothing per-timestep is ever stored, and that is
only possible if the loss decomposes over time.

A loss that reduces across time first — `jp.max_membrane_logits`, for instance — cannot be
trained this way and needs BPTT.

## The memory result

Peak scratch bytes for one gradient, 2-layer network:

| T | BPTT | e-prop | ratio |
|---:|---:|---:|---:|
| 100 | 210,984 | 3,128 | 68× |
| 1,000 | 2,090,024 | 3,128 | 668× |
| 4,000 | 8,354,024 | 3,128 | **2671×** |

The e-prop column is *identical* at every length. That is the whole point, and it is what makes
training on arbitrarily long streams possible.

## How the factorization works

For a weight from presynaptic unit `i` to postsynaptic unit `j`:

```
trace:            e_i[t] = alpha * e_i[t-1] + (1 - alpha) * s_i[t]
learning signal:  L_j[t] = dLoss / ds_j[t]           (spatial only, no time)
gradient:         dLoss/dW_ji = sum_t  L_j[t] * psi_j[t] * e_i[t]
```

where `psi` is the surrogate derivative. Note what the trace is: the presynaptic spike train
low-pass filtered by exactly the membrane's own time constant. That is not an approximation or
a coincidence — it is the membrane's impulse response, which is why the factorization works.

It is the same three-factor shape as reward-modulated STDP, with an error signal in place of
dopamine.

## How close the gradient is

Cosine similarity against the true BPTT gradient, 2-layer network, 40 timesteps:

| | reset-free (`LinearLIF`) | with reset (`LIF`) |
|---|---:|---:|
| layer feeding the loss directly | **1.000000** (exact to 2e-07) | 0.9988 |
| hidden layer | 0.879 | 0.917 |

Two independent sources of approximation, worth keeping apart:

**Reset.** A spike feeds back into its own membrane, a temporal path the factorization does not
carry. Without reset the membrane filter is the only route through time, and the gradient is
exact to float precision.

**Depth.** A hidden layer's learning signal would have to be filtered backwards through each
membrane to be exact — a backward pass in time, which is the thing online learning exists to
avoid. So it is propagated spatially only, the standard symmetric-feedback approximation. The
result is a well-aligned descent direction rather than the true gradient, and cosine near 0.9
is what makes it work as a learning rule despite not being exact.

## Scope

Currently supported: `Sequential` stacks of alternating `Dense` and spiking layers (`LIF`,
`LinearLIF`), optionally ending in a `LeakyIntegrator` readout. That is what the derivation
covers.

Anything else raises rather than silently returning a wrong gradient — in particular recurrent
`Graph`s, which are refused because recurrent paths are dropped by the same argument that drops
reset. Convolutional and recurrent support, along with OTTT and SLTT as comparison rules, are
not yet implemented.

## If you implement a learning rule of your own

Two details that are easy to get wrong, and neither fails loudly.

**The surrogate derivative belongs at the pre-reset membrane.** The threshold comparison
happens before reset, but a neuron returns post-reset state, so the pre-reset value must be
recomputed rather than read back. Evaluating it post-reset drops `LIF` gradient alignment from
0.9988 to 0.32, while leaving `LinearLIF` exact — without reset the two values coincide, so a
reset-free test suite will not catch it.

**Verify memory, not just the gradient.** An implementation can compute the correct gradient
and still stack activations across time, in which case memory grows with `T` and the method
delivers none of the benefit it exists for. Measure peak scratch at two sequence lengths and
confirm it is flat.
