"""Surrogate gradients.

A surrogate is defined by a *smooth relaxation* of the Heaviside step, and its gradient is
obtained from autodiff rather than being written out by hand. Defining a new surrogate means
writing one smooth function and nothing else: no hand-written derivative, no chance of forward
and backward disagreeing, and a gradient that can be finite-difference checked.

The forward pass emits the exact binary spike. The obvious way to write that is the
straight-through identity ``soft + stop_gradient(hard - soft)``, but it makes every forward
pass evaluate the relaxation only to cancel it -- and XLA cannot elide the work, because
float addition is not associative and `a + (b - a)` is not `b`. On a spiking network that is a
transcendental per neuron per timestep, thrown away.

`custom_jvp` removes it. The forward is a comparison; the tangent is obtained by
differentiating `relaxation` with `jax.jvp` at backward time. The derivative still comes from
autodiff -- subclasses gain nothing to get wrong -- it is simply no longer computed where it
is not needed.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.custom_jvp
def _spike(surrogate, v: Array) -> Array:
    return jnp.asarray(v > 0, v.dtype)


@_spike.defjvp
def _spike_jvp(primals, tangents):
    surrogate, v = primals
    d_surrogate, dv = tangents
    # jvp evaluates the relaxation as well as its derivative; the value is unused and XLA
    # eliminates it. Differentiating through `surrogate` too keeps learnable surrogate
    # parameters differentiable.
    _, tangent = jax.jvp(lambda s, x: s.relaxation(x), (surrogate, v), (d_surrogate, dv))
    return _spike(surrogate, v), tangent


class Surrogate(eqx.Module):
    """Base class. Subclasses implement `relaxation`."""

    def relaxation(self, v: Float[Array, "..."]) -> Float[Array, "..."]:
        """A smooth function of the membrane offset whose derivative is the surrogate."""
        raise NotImplementedError

    def __call__(self, v: Float[Array, "..."]) -> Float[Array, "..."]:
        return _spike(self, v)


class FastSigmoid(Surrogate):
    """Derivative ``1 / (1 + slope*|v|)^2``. Matches snnTorch's `fast_sigmoid`."""

    slope: float = 25.0

    def relaxation(self, v):
        return v / (1.0 + self.slope * jnp.abs(v)) + 0.5


class ATan(Surrogate):
    """Derivative ``(alpha/2) / (1 + (pi*alpha*v/2)^2)``. Matches snnTorch's `atan`."""

    alpha: float = 2.0

    def relaxation(self, v):
        return jnp.arctan(jnp.pi * self.alpha * v / 2.0) / jnp.pi + 0.5


class Triangular(Surrogate):
    """Derivative ``max(0, 1 - |v|/width) / width``. Compact support, cheapest backward."""

    width: float = 1.0

    def relaxation(self, v):
        # Written as u - u|u|/2 rather than sign(u)*(|u| - u^2/2): the two agree in value, but
        # jnp.sign has zero derivative everywhere, which would null the gradient at exactly
        # v == 0 -- the one point where the surrogate must be strongest.
        u = v / self.width
        a = jnp.abs(u)
        return 0.5 + jnp.where(a <= 1.0, u - 0.5 * u * a, jnp.sign(u) * 0.5)


class Boxcar(Surrogate):
    """Derivative ``1/width`` inside the window, zero outside. The straight-through estimator."""

    width: float = 1.0

    def relaxation(self, v):
        return jnp.clip(v / self.width, -0.5, 0.5) + 0.5


__all__ = ["ATan", "Boxcar", "FastSigmoid", "Surrogate", "Triangular"]
