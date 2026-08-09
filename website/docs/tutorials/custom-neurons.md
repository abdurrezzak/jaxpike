---
id: custom-neurons
title: Write your own neuron
sidebar_position: 2
---

# Write your own neuron

Neuron models are not a fixed menu. Anything satisfying a three-method contract is a neuron
here, with no registration, no subclassing and no special-casing — the execution engine, the
visualization tools and the training helpers all work on it unchanged.

This tutorial implements a neuron that does not ship with the library, verifies it, and makes
it eligible for parallel-in-time execution.

## The contract

```python
init_state(input_shape) -> state pytree
out_shape(input_shape)  -> output shape
__call__(state, x)      -> (new_state, output)
```

`__call__` returns state first, matching `jax.lax.scan`'s carry convention. State is an ordinary
pytree, so a neuron may carry one variable or several.

## A quadratic integrate-and-fire neuron

QIF is a standard model whose membrane accelerates as it approaches threshold, producing a
sharper spike onset than LIF:

```
v[t] = v[t-1] + dt·(v[t-1]·(v[t-1] - a) + x[t])
s[t] = H(v[t] - threshold)
v[t] = v[t] - threshold·s[t]
```

```python
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

import jaxpike as jp


class QIFState(eqx.Module):
    v: Float[Array, "..."]


class QIF(eqx.Module):
    """Quadratic integrate-and-fire."""

    a: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    surrogate: jp.Surrogate

    def __init__(self, *, a=0.5, threshold=1.0, dt=0.1, surrogate=None):
        self.a = a
        self.threshold = threshold
        self.dt = dt
        self.surrogate = surrogate if surrogate is not None else jp.ATan()

    def init_state(self, input_shape):
        return QIFState(v=jnp.zeros(input_shape))

    def out_shape(self, input_shape):
        return input_shape

    def __call__(self, state, x):
        v = state.v + self.dt * (state.v * (state.v - self.a) + x)
        s = self.surrogate(v - self.threshold)
        return QIFState(v=v - self.threshold * s), s
```

Two details worth copying.

**Hyperparameters are `static` fields.** A plain float would become a JAX leaf, which makes it
a traced value inside a `lax.scan` carry and changes the carry's structure between iterations.
Mark anything that is not a learned parameter as static.

**The surrogate is a submodule, not a function.** That keeps it swappable and, if it ever gains
learnable parameters, differentiable.

## Using it

It drops into `Sequential` like any built-in layer:

```python
import jax

k1, k2 = jax.random.split(jax.random.key(0))

net = jp.Sequential(
    jp.Dense(20, 64, key=k1),
    QIF(a=0.5, threshold=1.0, dt=0.1),
    jp.Dense(64, 2, key=k2),
    jp.LeakyIntegrator(tau=20.0),
)

xs = jax.random.uniform(jax.random.key(1), (50, 8, 20))
membrane, state = jp.unroll(net, xs)
```

## Verifying it

Three checks catch nearly every mistake in a new neuron.

**Does it fire at a sane rate?** A neuron that never spikes gives no gradient, and one that
always spikes carries no information:

```python
spikes, _ = jp.unroll(jp.Sequential(*net.layers[:2]), xs)
print(float(jp.density(spikes)))     # 0.013 for this configuration
```

Anything non-zero and below saturation works; 0.01–0.3 is the usual range. Exactly 0.0 is
fatal — the drive is too weak, so raise the input scale, lower the threshold, or pass `gain=`
to the preceding `Dense`. A rate near 1.0 is equally fatal in the other direction.

**Do gradients reach the first layer?** A silent or saturated neuron shows up as an exactly
zero gradient:

```python
import equinox as eqx

params, static = eqx.partition(net, eqx.is_inexact_array)
grads = jax.grad(lambda p: jnp.sum(jp.unroll(eqx.combine(p, static), xs)[0]))(params)
print(float(jnp.max(jnp.abs(grads.layers[0].weight))))    # must be non-zero
```

**Does state chunk correctly?** Feeding the returned state back in must equal the unchunked run.
This catches a state variable that was accidentally reset or dropped:

```python
whole, _ = jp.unroll(net, xs)
first, state = jp.unroll(net, xs[:25])
second, _ = jp.unroll(net, xs[25:], state)
assert jnp.allclose(whole, jnp.concatenate([first, second]))
```

## Making it parallel-in-time

`unroll_parallel` solves the whole time axis at once instead of stepping through it, but it
requires the recurrence to be **linear in the state**. QIF is quadratic and reset makes it
nonlinear again, so it does not qualify — and it must not pretend to.

A layer opts in by defining `parallel_apply(state, xs)`, and a layer that omits it is named in
the error rather than silently falling back. For a reset-free linear variant the implementation
is short:

```python
from jaxpike.parallel import scan_linear_recurrence


class LinearQIF(QIF):
    """Reset-free, linearized: v[t] = (1 + dt·(v0 - a))·v[t-1] + dt·x[t]."""

    def __call__(self, state, x):
        v = state.v + self.dt * (-self.a * state.v + x)
        return QIFState(v=v), self.surrogate(v - self.threshold)

    def parallel_apply(self, state, xs):
        decay = jnp.broadcast_to(1.0 - self.dt * self.a, xs.shape)
        v = scan_linear_recurrence(decay, self.dt * xs, state.v)
        return QIFState(v=v[-1]), self.surrogate(v - self.threshold)
```

Then assert the two paths agree, which is the test that matters:

```python
sequential, _ = jp.unroll(net, xs)
parallel, _ = jp.unroll_parallel(net, xs)
print(float(jnp.max(jnp.abs(sequential - parallel))))     # float32 noise, ~4e-8 here
```

Expect small disagreement rather than zero: the associative scan sums in a different order, and
float addition is not associative. What must match exactly is the binary spike train, since a
different rounding must never flip a threshold crossing.

## Writing a surrogate gradient

Surrogates follow the same principle. Define the smooth relaxation of the step function; the
derivative comes from autodiff, so forward and backward cannot disagree:

```python
class MySurrogate(jp.Surrogate):
    slope: float = 10.0

    def relaxation(self, v):
        return jax.nn.sigmoid(self.slope * v)
```

The forward pass emits an exact binary spike regardless — only the backward pass uses the
relaxation. Because the derivative is autodiff-consistent, it can be finite-difference tested:

```python
v = jnp.linspace(-1.0, 1.0, 5)
analytic = jax.grad(lambda x: jnp.sum(MySurrogate()(x)))(v)
numeric = (MySurrogate().relaxation(v + 1e-3) - MySurrogate().relaxation(v - 1e-3)) / 2e-3
print(jnp.max(jnp.abs(analytic - numeric)))
```

## See also

- [Neurons reference](../reference/neurons.md) — the models that ship with the library
- [Surrogates reference](../reference/surrogates.md) — the built-in relaxations
- [Execution](../guides/execution.md) — when each execution strategy applies
