"""Losses, readouts, and a training step.

The SNN-specific decision here is the **readout**: a spiking network emits a binary train over
time, and something has to turn that into class logits. Both options are provided because the
choice materially changes trainability:

- `count_logits` sums spikes per class. Interpretable, but the only gradient reaching it is
  the surrogate's, once per spike.
- `max_membrane_logits` takes the peak membrane potential, reading the *continuous* state
  before thresholding, so gradients flow even from units that never fired. Usually the better
  training target, and why the readout layer is normally left non-spiking.

`rate_penalty` covers the third choice: unconstrained SNNs drift toward firing at every
timestep or toward silence, and it pulls the mean rate back toward a target.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int


def count_logits(spikes: Float[Array, "T B C"]) -> Float[Array, "B C"]:
    """Class scores as total spike count over time."""
    return jnp.sum(spikes, axis=0)


def max_membrane_logits(membrane: Float[Array, "T B C"]) -> Float[Array, "B C"]:
    """Class scores as peak membrane potential over time.

    Preferred over spike counts for training: it is differentiable without passing through the
    threshold, so silent output units still receive gradient.
    """
    return jnp.max(membrane, axis=0)


def cross_entropy(logits: Float[Array, "B C"], labels: Int[Array, " B"]) -> Array:
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


def accuracy(logits: Float[Array, "B C"], labels: Int[Array, " B"]) -> Array:
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def rate_penalty(spikes: Float[Array, "T *rest"], target: float = 0.05) -> Array:
    """Squared deviation of the mean firing rate from `target`.

    Keeps a network off both failure modes: firing every step (no sparsity left to exploit)
    and firing never (no gradient left to learn from).
    """
    return (jnp.mean(spikes) - target) ** 2


def make_step(loss_fn, optimizer: optax.GradientTransformation):
    """Build a jitted `(model, opt_state, batch, labels) -> (model, opt_state, loss, aux)` step.

    `loss_fn(model, xs, labels)` must return `(loss, aux)`.
    """

    @eqx.filter_jit
    def step(model, opt_state, xs, labels):
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, xs, labels)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    return step


def iterate_batches(inputs, labels, batch_size: int, *, key, shuffle: bool = True):
    """Yield `(xs, ys)` batches. Inputs are `(N, T, ...)`; xs comes out time-major `(T, B, ...)`.

    **Keep `inputs` as a host (numpy) array.** Only the current batch is moved to the device;
    passing a device array pins the whole dataset in accelerator memory, which for a
    long-sequence spiking dataset is enormous (SHD at 1000 timesteps is 22.8 GB in float32).

    The trailing partial batch is dropped, which keeps every compiled step the same shape and
    avoids a recompile on the last batch of every epoch.
    """
    n = len(inputs)
    order = np.asarray(jax.random.permutation(key, n)) if shuffle else np.arange(n)
    for start in range(0, n - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        # Integer spike data is transferred in its narrow dtype and widened on the device,
        # which cuts PCIe traffic 4x versus converting to float32 on the host first.
        batch = np.swapaxes(inputs[idx], 0, 1)
        xs = jnp.asarray(batch)
        if not jnp.issubdtype(xs.dtype, jnp.floating):
            xs = xs.astype(jnp.float32)
        yield xs, jnp.asarray(labels[idx])


__all__ = [
    "accuracy",
    "count_logits",
    "cross_entropy",
    "iterate_batches",
    "make_step",
    "max_membrane_logits",
    "rate_penalty",
]
