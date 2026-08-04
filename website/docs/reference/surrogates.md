---
id: surrogates
title: Surrogate gradients
sidebar_position: 2
---

# Surrogate gradients

A spike is a Heaviside step, whose derivative is zero everywhere and undefined at threshold. A
surrogate gradient replaces that derivative with something usable in the backward pass while
keeping the forward pass exactly binary.

## Defined as relaxations, not as VJPs

In jaxpike a surrogate is defined by a **smooth relaxation** of the step, and its gradient comes
from autodiff:

```python
class MySurrogate(jp.Surrogate):
    slope: float = 10.0

    def relaxation(self, v):
        return jax.nn.sigmoid(self.slope * v)
```

`__call__` returns the exact binary spike forward and the relaxation's derivative backward, via
the straight-through identity `soft + stop_gradient(hard - soft)`.

Two consequences. Defining a surrogate means writing one smooth function and nothing else — no
custom VJP, no chance of forward and backward disagreeing. And because the derivative is
autodiff-consistent with `relaxation`, it can be **finite-difference checked**, which a
hand-written surrogate derivative cannot be.

This catches errors that are otherwise invisible. A `Triangular` relaxation written on
`jnp.sign` returns *zero gradient exactly at threshold* — the one point where the surrogate
must be strongest — because `jnp.sign` has zero derivative everywhere. The implementation here
uses `u - u|u|/2`, which agrees in value and has the right derivative.

The cost is that the forward pass evaluates the relaxation even though its value is discarded.
That is a few elementwise ops.

## Built in

All take their parameter as a keyword with the default shown.

| Surrogate | Derivative | Notes |
|---|---|---|
| `jp.FastSigmoid(slope=25.0)` | `1 / (1 + slope·abs(v))^2` | the default; matches snnTorch's `fast_sigmoid` |
| `jp.ATan(alpha=2.0)` | `(alpha/2) / (1 + (pi·alpha·v/2)^2)` | matches snnTorch's `atan` |
| `jp.Triangular(width=1.0)` | `max(0, 1 - abs(v)/width) / width` | compact support, cheapest backward |
| `jp.Boxcar(width=1.0)` | `1/width` inside the window, 0 outside | the straight-through estimator |

```python
jp.LIF(tau=20.0, surrogate=jp.ATan(alpha=2.0))
```

`FastSigmoid()` is used when you pass nothing.

## Choosing one

The choice matters less than the scale. A surrogate with a narrow effective width gives no
gradient to neurons sitting far from threshold, which is the same silence problem as a badly
initialized network — see [Why deep SNNs go silent](../guides/silent-networks.md). If training
stalls, widening the surrogate (lower `slope`, higher `width`) is worth trying before anything
more elaborate.

`Boxcar` is the most aggressive and the least smooth; `ATan` and `FastSigmoid` have long tails,
so distant neurons still receive a small gradient.

## Where the derivative is evaluated

At the **pre-reset** membrane, always. The threshold comparison happens before reset, so that is
where `psi` belongs. This is invisible for reset-free neurons, where the two membranes coincide,
and it is a large error for `LIF`: evaluating it post-reset dropped e-prop's `LIF` gradient
alignment from 0.9988 to 0.32. If you are implementing a learning rule that needs `psi`
explicitly, recompute the pre-reset membrane rather than reading the returned state.
