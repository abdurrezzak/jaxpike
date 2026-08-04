"""Surrogate gradients.

A surrogate is defined by a *smooth relaxation* of the Heaviside step, and its gradient is
obtained from autodiff rather than being written out by hand. `__call__` returns the exact
binary spike in the forward pass and the relaxation's derivative in the backward pass, via
the straight-through identity ``soft + stop_gradient(hard - soft)``.

Defining a new surrogate therefore means writing one smooth function and nothing else: no
custom VJP, no chance of forward and backward disagreeing, and a derivative that can be
finite-difference checked.

The cost is that the forward pass evaluates the relaxation even though its value is
discarded, which is a few elementwise ops.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Surrogate(eqx.Module):
    """Base class. Subclasses implement `relaxation`."""

    def relaxation(self, v: Float[Array, "..."]) -> Float[Array, "..."]:
        """A smooth function of the membrane offset whose derivative is the surrogate."""
        raise NotImplementedError

    def __call__(self, v: Float[Array, "..."]) -> Float[Array, "..."]:
        hard = (v > 0).astype(v.dtype)
        soft = self.relaxation(v)
        return soft + jax.lax.stop_gradient(hard - soft)


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
