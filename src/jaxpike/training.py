"""Losses, readouts, and a training step.

The one genuinely SNN-specific decision here is the **readout**: a spiking network emits a
binary train over time, and something has to turn that into class logits. Two options, both
provided, because the choice materially changes trainability:

- `count_logits` sums spikes per class. Interpretable, and what the accuracy metric ultimately
  reflects, but the gradient reaching it is the surrogate's, once per spike.
- `max_membrane_logits` takes the peak membrane potential instead. It reads the *continuous*
  state before thresholding, so gradients flow even from units that never fired — which is
  exactly the case where a spike-count readout gives no signal at all. This is usually the
  better training target, and it is why the readout layer is normally left non-spiking.

A third choice, and the reason the API takes logits rather than a network: regularizing the
firing rate. Unconstrained SNNs drift toward firing at every timestep (which wastes the
sparsity that makes them interesting) or toward silence (which kills the gradient), so
`rate_penalty` pulls the mean rate toward a target.
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

    **Keep `inputs` as a host (numpy) array.** Only the current batch is moved to the device.
    Passing a device array instead pins the whole dataset in accelerator memory, which for a
    long-sequence spiking dataset is enormous: SHD at 1000 timesteps is 8156 x 1000 x 700
    float32 = 22.8 GB, more than most GPUs have, before the model allocates anything at all.

    The trailing partial batch is dropped, which keeps every compiled step the same shape and
    avoids a recompile on the last batch of every epoch.
    """
    n = len(inputs)
    order = np.asarray(jax.random.permutation(key, n)) if shuffle else np.arange(n)
    for start in range(0, n - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        # Time-major on the host, then a single transfer of just this batch. Integer spike
        # data is transferred in its narrow dtype and widened on the device, which cuts PCIe
        # traffic 4x versus converting to float32 first.
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
