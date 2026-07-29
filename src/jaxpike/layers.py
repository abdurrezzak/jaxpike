"""Stateless connection layers and the container that threads state through them."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Dense(eqx.Module):
    """Affine map over the trailing axis. Batch axes are arbitrary and untouched."""

    weight: Float[Array, "out in"]
    bias: Float[Array, " out"] | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        use_bias: bool = True,
    ):
        # LeCun normal: variance 1/fan_in keeps pre-activation scale stable at init, which for
        # a spiking net is what sets the initial firing rate.
        lim = jnp.sqrt(1.0 / in_features)
        self.weight = lim * jax.random.normal(key, (out_features, in_features))
        self.bias = jnp.zeros((out_features,)) if use_bias else None

    def init_state(self, input_shape: tuple[int, ...]) -> None:
        return None

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        if input_shape[-1] != self.weight.shape[1]:
            raise ValueError(
                f"Dense expects trailing dim {self.weight.shape[1]}, got shape {input_shape}"
            )
        return (*input_shape[:-1], self.weight.shape[0])

    def __call__(self, state: None, x: Array) -> tuple[None, Array]:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return None, y


class Sequential(eqx.Module):
    """Applies layers in order at each timestep, carrying each layer's state separately.

    State is a tuple parallel to `layers`, with `None` for stateless ones. Keeping it as a
    tuple rather than a flat array is what lets `lax.scan` carry heterogeneous neuron states
    (LIF's one variable, ALIF's two) without special-casing.
    """

    layers: tuple

    def __init__(self, *layers):
        self.layers = tuple(layers)

    def init_state(self, input_shape: tuple[int, ...]) -> tuple:
        states, shape = [], input_shape
        for layer in self.layers:
            states.append(layer.init_state(shape))
            shape = layer.out_shape(shape)
        return tuple(states)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        shape = input_shape
        for layer in self.layers:
            shape = layer.out_shape(shape)
        return shape

    def __call__(self, state: tuple, x: Array) -> tuple[tuple, Array]:
        new_state = []
        for layer, s in zip(self.layers, state, strict=True):
            s, x = layer(s, x)
            new_state.append(s)
        return tuple(new_state), x


__all__ = ["Dense", "Sequential"]
