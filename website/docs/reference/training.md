---
id: training
title: Training
sidebar_position: 5
---

# Training

## Readouts: the one genuinely SNN-specific choice

A spiking network emits a binary train over time, and something has to turn that into class
logits. Both options are provided because the choice materially changes trainability.

```python
jp.count_logits(spikes)            # sum of spikes per class over time
jp.max_membrane_logits(membrane)   # peak membrane potential per class over time
```

`count_logits` is interpretable and is what the accuracy metric ultimately reflects, but the
only gradient reaching it is the surrogate's, once per spike. **A class that never fires gets no
signal at all and can never learn to fire.**

`max_membrane_logits` reads the *continuous* state before thresholding, so gradients flow even
from units that never fired. This is usually the better training target, and it is why the
readout layer is normally a non-spiking `jp.LeakyIntegrator`.

## Losses

```python
jp.cross_entropy(logits, labels)     # mean softmax cross-entropy, integer labels
jp.accuracy(logits, labels)          # mean argmax accuracy
jp.rate_penalty(spikes, target=0.05) # squared deviation of the mean firing rate from target
```

`rate_penalty` keeps a network off both failure modes: firing every step, which wastes the
sparsity that makes SNNs interesting, and firing never, which kills the gradient.

```python
loss = jp.cross_entropy(logits, labels) + 0.1 * jp.rate_penalty(hidden_spikes, 0.05)
```

## `make_step`

```python
step = jp.make_step(loss_fn, optimizer)
model, opt_state, loss, aux = step(model, opt_state, xs, labels)
```

Builds a jitted step from a `loss_fn(model, xs, labels) -> (loss, aux)` and an optax
transformation. It wraps `eqx.filter_value_and_grad(..., has_aux=True)`, applies the optimizer
update, and returns the new model and optimizer state. `aux` is whatever the second element of
your loss tuple was — accuracy, typically.

For [e-prop](../guides/online-learning.md), replace `eqx.filter_value_and_grad` with
`jp.eprop_value_and_grad` and drive the optimizer yourself.

## `iterate_batches`

```python
for xs, ys in jp.iterate_batches(inputs, labels, batch_size, *, key, shuffle=True):
    ...
```

Inputs are `(N, T, ...)` on the host; `xs` comes out **time-major** `(T, B, ...)` on the device,
which is the layout every `unroll` variant expects.

**Keep `inputs` as a host (NumPy) array.** Only the current batch is moved to the device.
Passing a device array pins the whole dataset in accelerator memory, which for a long-sequence
spiking dataset is enormous: SHD at 1000 timesteps is 8156 × 1000 × 700 float32 = 22.8 GB, more
than most GPUs have, before the model allocates anything.

Integer spike data is transferred in its narrow dtype and widened on the device, which cuts PCIe
traffic 4× against converting to float32 first. That single change moved a long-sequence
benchmark from an apparent 1.56× speedup to a true 2.4×, because the input pipeline had been the
bottleneck.

The trailing partial batch is dropped, which keeps every compiled step the same shape and avoids
a recompile on the last batch of every epoch.

## A complete loop

```python
import equinox as eqx
import jax
import optax
import jaxpike as jp

runner = jp.unroll_parallel

def loss_fn(m, xs, labels):
    membrane, _ = runner(m, xs)
    logits = jp.max_membrane_logits(membrane)
    return jp.cross_entropy(logits, labels), jp.accuracy(logits, labels)

optimizer = optax.adamw(2e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
step = jp.make_step(loss_fn, optimizer)

for epoch in range(epochs):
    key = jax.random.key(epoch)
    for xs, ys in jp.iterate_batches(x_train, y_train, 128, key=key):
        model, opt_state, loss, acc = step(model, opt_state, xs, ys)
```

## Evaluating honestly

Select the epoch on a held-out validation split, never on test. Reporting the best test accuracy
across epochs selects the epoch *on the test set*; measured here, that inflated results by 0.5
to 4 points. The worked example is in [Train SHD end to end](../getting-started/training-shd.md).
