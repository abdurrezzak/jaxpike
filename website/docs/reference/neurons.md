---
id: neurons
title: Neurons
sidebar_position: 1
---

# Neurons

Every neuron in `jaxpike.neurons` is the correctness reference for its model: pure JAX, no
kernels, no tricks. Clarity matters more than speed here, because everything faster is
validated against these.

## The state contract

Any module with these three methods works everywhere in the library — in `Sequential`, in
`Graph`, in every `unroll` variant, and in the visualization functions. Nothing is registered,
subclassed or special-cased.

```python
init_state(input_shape) -> state pytree
out_shape(input_shape)  -> output shape
__call__(state, x)      -> (new_state, spikes)
```

State is explicit and functional. Nothing is stored on the module between steps.

Optionally, a neuron may implement `parallel_apply(state, xs) -> (final_state, outputs)` to
opt into [parallel-in-time execution](../guides/execution.md). Only reset-free models can.

**Membrane state is always float32**, even under bf16 or fp16 training. A leaky integrator runs
for thousands of steps, and low-precision accumulation drifts enough to change which neurons
cross threshold.

## `LIF`

```python
jp.LIF(tau=20.0, *, threshold=1.0, dt=1.0, reset="subtract", surrogate=None)
```

```
v[t] = alpha*v[t-1] + (1 - alpha)*x[t],   alpha = exp(-dt/tau)
s[t] = H(v[t] - threshold)
```

followed by a reset: `"subtract"` removes one threshold from the membrane, retaining the
overshoot so information is not discarded; `"zero"` clamps it back to rest.

**Note the `(1 - alpha)` on the input.** This is the normalized convention: a constant drive `x`
drives the membrane to a steady state of exactly `x`, so inputs are expressed in the same units
as `threshold` and a drive below threshold provably never fires. The practical consequence is
that a single timestep injects only `1 - alpha` of the input — about 5% at `tau=20` — so
transient drives must be large to fire on their own. snnTorch omits this factor; see
[Coming from snnTorch](../guides/coming-from-snntorch.md).

`tau` is stored as `log_tau` so it stays positive under unconstrained optimization, and it is a
**learnable leaf by default**. Freeze it with `equinox.partition` if you don't want that.
`tau` and `alpha` are exposed as properties. Construction raises if `tau <= dt`, where the
discretization would be unstable.

State: `LIFState(v)`.

## `LinearLIF`

```python
jp.LinearLIF(tau=20.0, *, threshold=1.0, dt=1.0, surrogate=None)
```

Identical to `LIF` except that a spike does not perturb the membrane — there is no reset, and
therefore no `reset` argument.

Dropping the reset is a real modelling choice with a real cost and a real payoff. The cost: the
neuron cannot regulate its own firing, so a strongly driven unit saturates at one spike per
timestep. The payoff: the recurrence stays affine, so the whole time axis can be solved with an
associative scan — 119× faster than sequential at `T=8192` on a T4 for an isolated membrane.

This is the PSN-style neuron of the parallel spiking network literature, and it is the tier-1
case for `unroll_parallel`. It implements `parallel_apply`.

State: `LIFState(v)`.

## `LeakyIntegrator`

```python
jp.LeakyIntegrator(tau=20.0, *, dt=1.0)
```

A LIF that never spikes: outputs membrane potential directly.

This is the standard SNN readout layer, and choosing it over spike counting matters. Classifying
on spike counts means the loss only sees a unit once it crosses threshold, so a class that never
fires produces no gradient and can never learn to fire. Reading the continuous membrane keeps
every output unit differentiable from the first step. Pair it with `jp.max_membrane_logits`.

Being linear and reset-free, it also implements `parallel_apply`.

State: `LIFState(v)`.

## `ALIF`

```python
jp.ALIF(tau=20.0, tau_a=200.0, *, beta=1.8, threshold=1.0, dt=1.0, reset="subtract",
        surrogate=None)
```

Adaptive LIF: each spike raises the neuron's own threshold, which then decays. The effective
threshold is `threshold + beta*a`, where the adaptation variable `a` integrates the neuron's own
spikes with time constant `tau_a`.

This is the smallest model that needs two state variables, so it is the worked example if you
are writing your own multi-variable neuron.

State: `ALIFState(v, a)`. No `parallel_apply`, and it cannot be exported to NIR.

## `Izhikevich`

```python
jp.Izhikevich(a=0.02, b=0.2, c=-65.0, d=8.0, *, v_peak=30.0, v_scale=5.0, dt=1.0,
              substeps=2, surrogate=None)
jp.Izhikevich.preset("chattering")
```

```
v' = 0.04*v^2 + 5*v + 140 - u + I
u' = a*(b*v - u)
if v >= 30 mV:  v <- c,  u <- u + d
```

Unlike LIF this is a *spike-generating* model rather than a threshold-crossing one: the
quadratic term produces a genuine upstroke, and 30 mV is where the spike is detected at its peak
rather than a threshold in the LIF sense. That is what buys the variety — bursting, chattering,
adaptation, rebound — from four parameters.

<img src="/img/figures/izhikevich_light.png" alt="Izhikevich firing patterns" className="figure-light" />
<img src="/img/figures/izhikevich_dark.png" alt="Izhikevich firing patterns" className="figure-dark" />

Presets in `jp.IZHIKEVICH_PRESETS`:

| Name | a | b | c | d |
|---|---:|---:|---:|---:|
| `regular_spiking` | 0.02 | 0.20 | −65 | 8.0 |
| `intrinsically_bursting` | 0.02 | 0.20 | −55 | 4.0 |
| `chattering` | 0.02 | 0.20 | −50 | 2.0 |
| `fast_spiking` | 0.10 | 0.20 | −65 | 2.0 |
| `low_threshold_spiking` | 0.02 | 0.25 | −65 | 2.0 |
| `resonator` | 0.10 | 0.26 | −65 | 2.0 |
| `thalamo_cortical` | 0.02 | 0.25 | −65 | 0.05 |

Three practical notes, because this model is less forgiving than LIF:

**Voltages are in millivolts**, not the dimensionless units the LIF classes use. Resting
potential is around −65 mV and input currents are of order 1–20. Weight initialization tuned
for LIF — including `lif_gain` — does nothing useful here.

**The quadratic term makes forward Euler unstable at `dt=1`**: `v` can run away to infinity
within a step under large input. Sub-stepping (two half steps, as Izhikevich's own code does)
plus a clamp above the detection level keeps it finite without changing the dynamics below the
spike.

**It is nonlinear in time**, so there is no `parallel_apply` and `unroll_parallel` will say so.

State: `IzhikevichState(v, u)`.

## Writing your own

Implement the three-method contract and you are done. `ALIF` is the reference for a
two-variable model. If your neuron's recurrence is affine in the membrane, add `parallel_apply`
using `jaxpike.parallel.scan_linear_recurrence` and it will run parallel-in-time:

```python
def parallel_apply(self, state, xs):
    from jaxpike.parallel import scan_linear_recurrence
    alpha = self.alpha
    a = jnp.broadcast_to(alpha, xs.shape)
    b = (1.0 - alpha) * xs.astype(jnp.float32)
    v = scan_linear_recurrence(a, b, state.v)
    return LIFState(v=v[-1]), self.surrogate(v - self.threshold)
```

One pitfall: decay factors computed from static Python floats must use `math.exp`, not
`jnp.exp`. `float(jnp.exp(...))` on a static value raises inside a jitted scan.
