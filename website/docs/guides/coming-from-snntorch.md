---
id: coming-from-snntorch
title: Coming from snnTorch
sidebar_position: 1
---

# Coming from snnTorch

Most of the translation is mechanical. Two things are not, and both change your numbers
silently rather than raising: the **input normalization convention** and the **reset timing**.
Read those two sections even if you skim the rest.

Everything asserted here is checked in `tests/test_cross_framework.py`, which runs against real
snnTorch and Norse when the `[bench]` extra is installed.

## The two numerical differences

### 1. Weights need rescaling by `1 / (1 - alpha)`

jaxpike uses the normalized input convention:

```
jaxpike:   v[t] = alpha*v[t-1] + (1 - alpha)*x[t]
snnTorch:  m[t] = beta*m[t-1]  + x[t]
```

with `alpha = beta = exp(-dt/tau)`. The decay term is identical. The difference is the
`(1 - alpha)` factor on the input, and it is a factor of about 20 at `tau=20`.

The reason: a constant drive `x` settles at a steady state of exactly `x`, so inputs are
expressed in the same units as `threshold`, and a drive below threshold provably never fires.
The practical cost is that a single timestep injects only about 5% of the input at `tau=20`, so
a transient must be large to fire on its own.

**Porting a trained snnTorch model:** multiply the incoming weights by `1 / (1 - alpha)`.

```python
import math

alpha = math.exp(-1.0 / 20.0)      # 0.9512
scale = 1.0 / (1.0 - alpha)        # 20.5
```

With that one correction the subthreshold traces agree to float32 precision — that is the
content of `test_snntorch_matches_ours_when_input_is_rescaled`, which asserts a worst deviation
below `1e-5`.

### 2. Reset is applied before the next decay, not after

```
jaxpike:   v[t] = a*(v[t-1] - thr*s[t-1]) + (1-a)*x[t]     reset is attenuated by a
snnTorch:  m[t] = a*m[t-1] - thr*s[t-1] + x[t]             reset is applied at full size
```

Both appear in the literature and both are defensible; they are not equivalent. The difference
is bounded and asserted in `test_reset_timing_differs_from_snntorch_and_the_difference_is_bounded`,
so it is a documented fact rather than a surprise. There is no flag to switch it — if you need
snnTorch's exact discretization, write a neuron that implements it; the state contract is three
methods.

## API translation

| snnTorch | jaxpike |
|---|---|
| `snn.Leaky(beta=0.95)` | `jp.LIF(tau=20.0)` — `beta = exp(-dt/tau)`, so `tau = -dt/log(beta)` |
| `snn.Leaky(..., reset_mechanism="subtract")` | `jp.LIF(reset="subtract")` (the default) |
| `snn.Leaky(..., reset_mechanism="zero")` | `jp.LIF(reset="zero")` |
| `snn.RLeaky` | a [`Graph`](./topologies.md) with a back-edge |
| `snn.Synaptic` | no equivalent yet (current-based synapse) |
| `snn.Alpha` | no equivalent yet |
| adaptive threshold | `jp.ALIF(tau=20.0, tau_a=200.0, beta=...)` |
| `nn.Linear` | `jp.Dense(in, out, key=...)` |
| `nn.Conv2d` | `jp.Conv2d(in_ch, out_ch, k, key=...)` — **NHWC**, not NCHW |
| `surrogate.fast_sigmoid(slope)` | `jp.FastSigmoid(slope=25.0)` |
| `surrogate.atan(alpha)` | `jp.ATan(alpha=2.0)` |
| manual `for t in range(T)` loop | `jp.unroll(net, xs)` |
| `mem = torch.zeros(...)` before the loop | nothing — `unroll` calls `init_state` for you |

### `beta` versus `tau`

snnTorch parameterizes the decay directly; jaxpike parameterizes the time constant. Convert
with:

```python
import math
tau = -1.0 / math.log(beta)        # beta=0.95 -> tau=19.5
beta = math.exp(-1.0 / tau)        # tau=20    -> beta=0.9512
```

`log(tau)` is stored internally so it stays positive under unconstrained optimization, and `tau`
is a learnable leaf by default. That differs from snnTorch, where `beta` is a buffer unless you
pass `learn_beta=True`. To freeze it, partition the model with `equinox.partition` on the
`log_tau` leaves.

## Shape and layout differences

**Time is the leading axis: `(time, batch, features)`.** snnTorch loops are usually written with
the batch outermost and time supplied by your own loop. `unroll` scans the leading axis.

**Convolutions are NHWC**: `(time, batch, height, width, channels)`. XLA's convolutions are
written for channels-last, and NCHW forces a transpose around every op. If you are porting conv
weights, transpose them.

## Structural differences

**No `nn.Module` state.** Networks are immutable pytrees. There is no `mem` attribute mutated in
place; `unroll` threads state through a `lax.scan` and returns the final state. This is what
makes truncated BPTT a one-liner:

```python
spikes_a, state = jp.unroll(net, xs[:50])
spikes_b, state = jp.unroll(net, xs[50:], state)   # exactly equals the unchunked run
```

**No `utils.reset(net)`.** State is created by `init_state` on every `unroll` call unless you
pass one in, so there is nothing to forget to reset.

**Surrogates are relaxations, not custom autograd functions.** You write the smooth function and
autodiff differentiates it, so forward and backward cannot disagree. See
[Surrogate gradients](../reference/surrogates.md).

## What jaxpike has that snnTorch does not

- **Parallel-in-time execution** for reset-free neurons, on long sequences. See
  [Execution](./execution.md).
- **Rematerialized BPTT** — 67× less memory at `T=5000`, one function call.
- **e-prop** with memory flat in sequence length. See [Online learning](./online-learning.md).
- **Arbitrary topologies** as data, via [`Graph`](./topologies.md).
- **An exact LIF integrator.** The membrane is the closed-form ODE solution, verified against
  the analytic `x*(1 - exp(-t/tau))`; Norse's forward Euler carries `O(dt/tau)` truncation
  error, and the tests measure the gap and confirm both converge as `dt` shrinks.

## What snnTorch has that jaxpike does not

Tutorials and community, a larger neuron zoo (`Synaptic`, `Alpha`, and more), and a PyTorch
ecosystem you may already depend on. If you want a spiking model inside an existing PyTorch
training stack, staying there is the right call.

## Moving a model between the two

Use [NIR](./nir.md). Two caveats specific to snnTorch, both verified:

- snnTorch's NIR importer assumes `dt = 1e-4 s` regardless of what the file says.
- It has no mapping for NIR's `LI` node, so a `LeakyIntegrator` readout will not cross into it.

Round-tripping *within* jaxpike is exact. Leaving the library is not bit-exact, because NIR
specifies a differential equation rather than a discretization. Check numerically on the far
side.
