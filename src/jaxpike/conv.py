"""Convolutional and pooling layers.

**Layout is NHWC**, i.e. `(time, batch, height, width, channels)`. PyTorch users will expect
NCHW, but XLA's convolutions are written for channels-last and NCHW forces layout transposes
around every op on GPU and TPU.

Every layer here is stateless and applied independently at each timestep, so all of them
parallelize over time for free: fold the time axis into the batch, run one big convolution,
unfold.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)


def _spatial_out(size: int, kernel: int, stride: int, padding: str) -> int:
    if padding == "SAME":
        return -(-size // stride)  # ceil division
    if padding == "VALID":
        return -(-(size - kernel + 1) // stride)
    raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding!r}")


class _TimeFolded(eqx.Module):
    """Mixin for stateless per-timestep layers.

    `__call__` handles one timestep; `parallel_apply` folds the leading time axis into the
    batch so the whole sequence goes through a single op. Since the layer holds no state,
    the two are guaranteed to agree.
    """

    def init_state(self, input_shape: tuple[int, ...]) -> None:
        return None

    def parallel_apply(self, state: None, xs: Array) -> tuple[None, Array]:
        t, batch = xs.shape[0], xs.shape[1]
        folded = xs.reshape(t * batch, *xs.shape[2:])
        _, out = self(None, folded)
        return None, out.reshape(t, batch, *out.shape[1:])


class Conv2d(_TimeFolded):
    """2D convolution over `(batch, height, width, channels)`, applied per timestep."""

    weight: Float[Array, "kh kw in out"]
    bias: Float[Array, " out"] | None
    stride: tuple[int, int] = eqx.field(static=True)
    padding: str = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        *,
        key: PRNGKeyArray,
        stride: int | tuple[int, int] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        gain: float = 1.0,
    ):
        kh, kw = _pair(kernel_size)
        if padding not in ("SAME", "VALID"):
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding!r}")
        # LeCun normal over the true fan-in; see Dense for why `gain` matters in a spiking net.
        fan_in = kh * kw * in_channels
        self.weight = (
            gain
            * jnp.sqrt(1.0 / fan_in)
            * jax.random.normal(key, (kh, kw, in_channels, out_channels))
        )
        self.bias = jnp.zeros((out_channels,)) if use_bias else None
        self.stride = _pair(stride)
        self.padding = padding

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        *lead, h, w, c = input_shape
        kh, kw, in_c, out_c = self.weight.shape
        if c != in_c:
            raise ValueError(f"Conv2d expects {in_c} input channels, got shape {input_shape}")
        return (
            *lead,
            _spatial_out(h, kh, self.stride[0], self.padding),
            _spatial_out(w, kw, self.stride[1], self.padding),
            out_c,
        )

    def __call__(self, state: None, x: Array) -> tuple[None, Array]:
        y = jax.lax.conv_general_dilated(
            x,
            self.weight,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias is not None:
            y = y + self.bias
        return None, y


class Pool2d(_TimeFolded):
    """Spatial pooling. `mode` is ``"max"`` or ``"avg"``.

    Average pooling is usually the right default in an SNN: inputs are binary spikes, so max
    pooling over a window saturates to 1 as soon as any unit in it fires and discards how many
    did. Average pooling keeps that count, which is the information the next layer needs.
    """

    window: tuple[int, int] = eqx.field(static=True)
    stride: tuple[int, int] = eqx.field(static=True)
    mode: str = eqx.field(static=True)

    def __init__(
        self,
        window: int | tuple[int, int] = 2,
        *,
        stride: int | tuple[int, int] | None = None,
        mode: str = "avg",
    ):
        if mode not in ("max", "avg"):
            raise ValueError(f"mode must be 'max' or 'avg', got {mode!r}")
        self.window = _pair(window)
        self.stride = _pair(stride) if stride is not None else self.window
        self.mode = mode

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        *lead, h, w, c = input_shape
        return (
            *lead,
            _spatial_out(h, self.window[0], self.stride[0], "VALID"),
            _spatial_out(w, self.window[1], self.stride[1], "VALID"),
            c,
        )

    def __call__(self, state: None, x: Array) -> tuple[None, Array]:
        window = (1, *self.window, 1)
        strides = (1, *self.stride, 1)
        if self.mode == "max":
            y = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, window, strides, "VALID")
        else:
            summed = jax.lax.reduce_window(x, 0.0, jax.lax.add, window, strides, "VALID")
            y = summed / (self.window[0] * self.window[1])
        return None, y


class Flatten(_TimeFolded):
    """Collapse all trailing axes into one, preserving the batch axis."""

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        batch, *rest = input_shape
        size = 1
        for dim in rest:
            size *= dim
        return (batch, size)

    def __call__(self, state: None, x: Array) -> tuple[None, Array]:
        return None, x.reshape(x.shape[0], -1)


__all__ = ["Conv2d", "Flatten", "Pool2d"]
