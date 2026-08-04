---
id: training-shd
title: Train SHD end to end
sidebar_position: 3
---

# Train SHD end to end

The Spiking Heidelberg Digits are the field's standard temporal benchmark: 20 classes (digits
0–9 spoken in English and German), converted to spike trains over 700 input channels by a
cochlea model. Unlike rate-coded MNIST the temporal structure is real, which is what makes it
worth training on.

Reference accuracies from Cramer et al. (2020), who introduced the dataset: **~48% feedforward,
~71% recurrent**. This page reproduces 0.626 feedforward and 0.696 recurrent.

The finished script is `examples/shd.py` in the repository. This page builds it up.

## 1. Data

```python
import jaxpike as jp

train = jp.data.shd("train", "data/shd", timesteps=250)
test = jp.data.shd("test", "data/shd", timesteps=250)
print(train)
# Dataset(shd/train: 8156 samples, 250 timesteps, 700 units, 20 classes, ...)
```

The loader downloads the HDF5 files on first use and bins the event lists into dense arrays.
Needs the `[data]` extra (h5py).

Two deliberate choices, both of which affect whether long-sequence training fits and runs at
speed:

**Arrays stay on the host as NumPy, not on the device as JAX arrays.** SHD at `timesteps=1000`
is 8156 × 1000 × 700, which is 22.8 GB in float32 — enough to exhaust a 16 GB accelerator
before training starts. `jp.iterate_batches` moves one batch at a time.

**Spikes are stored as `uint8`.** They are binary, so float32 costs four times the memory and
four times the host-to-device bandwidth per batch. On long sequences that is the difference
between being compute-bound and transfer-bound, and it is large enough to mask a real speedup
in benchmarking.

## 2. Model

```python
import jax

N_CHANNELS, N_CLASSES = 700, 20
tau, threshold, hidden = 20.0, 0.5, 256

gain = jp.lif_gain(tau)            # 6.33 at tau=20 -- see the "silent networks" guide
k1, k2, k3 = jax.random.split(jax.random.key(0), 3)

model = jp.Sequential(
    jp.Dense(N_CHANNELS, hidden, key=k1, gain=gain),
    jp.LinearLIF(tau=tau, threshold=threshold),
    jp.Dense(hidden, hidden, key=k2, gain=gain),
    jp.LinearLIF(tau=tau, threshold=threshold),
    jp.Dense(hidden, N_CLASSES, key=k3, gain=gain),
    jp.LeakyIntegrator(tau=tau),
)
```

`LinearLIF` is reset-free, which means the whole stack runs parallel-in-time. The readout is a
`LeakyIntegrator` so every output unit is differentiable from the first timestep.

`gain=jp.lif_gain(tau)` is not optional at this depth. Without it the third layer fires at
exactly zero and the network has no gradient anywhere.

## 3. Loss and training step

```python
import equinox as eqx
import optax

runner = jp.unroll_parallel                    # jp.unroll for the sequential path

def loss_fn(m, xs, labels):
    membrane, _ = runner(m, xs)
    logits = jp.max_membrane_logits(membrane)
    return jp.cross_entropy(logits, labels), jp.accuracy(logits, labels)

optimizer = optax.adamw(2e-3)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
step = jp.make_step(loss_fn, optimizer)
```

If the hidden layers saturate, add `jp.rate_penalty(hidden_spikes, target=0.05)` to the loss.
It is off by default here because it was not needed.

## 4. Honest evaluation

This part matters more than the architecture. **Never report the best test accuracy across
epochs.** Doing so selects the epoch on the test set, which inflates the figure — measured on
this benchmark, by 0.5 to 4 points depending on the run.

```python
import numpy as np

rng = np.random.default_rng(0)
order = rng.permutation(len(train.labels))
n_val = int(0.1 * len(order))
val_idx, train_idx = order[:n_val], order[n_val:]
```

Select the epoch on the validation split, then report test once at the selected epoch.

The protocol is not pedantry here; it changes what you learn from the run. The feedforward
model scores *higher* on validation and *lower* on test than the recurrent one, because SHD's
test set holds out entire speakers. Recurrence buys speaker generalization rather than raw
capacity — a distinction that selecting on test hides completely.

## 5. The loop

```python
for epoch in range(epochs):
    key = jax.random.key(epoch + 1)
    for xs, ys in jp.iterate_batches(x_train, y_train, batch_size=128, key=key):
        model, opt_state, loss, acc = step(model, opt_state, xs, ys)
    val_acc = evaluate_all(model, x_val, y_val, key)
    test_acc = evaluate_all(model, x_test, y_test, key)
```

`iterate_batches` shuffles on the host and transfers one batch at a time, which is what keeps
the uint8 storage decision paying off.

## 6. Running it

```bash
python examples/shd.py --epochs 20
python examples/shd.py --epochs 100 --recurrent --augment      # the 0.696 run
```

On a GPU through Modal:

```bash
python -m modal run benchmarks/gpu/run_modal.py --bench shd \
    --extra "--epochs 100 --recurrent --augment"
```

## 7. The recurrent variant

The recurrent model is a [`Graph`](../guides/topologies.md), because the hidden layer feeds
itself:

```python
model = jp.Graph(
    nodes={
        "w_in": jp.Dense(N_CHANNELS, hidden, key=k1, gain=gain),
        "h1": jp.LIF(tau=tau, threshold=threshold, reset="subtract"),
        "w_rec": jp.Dense(hidden, hidden, key=k2, gain=0.2),     # note the much smaller gain
        "w_h2": jp.Dense(hidden, hidden, key=k3, gain=gain),
        "h2": jp.LIF(tau=tau, threshold=threshold, reset="subtract"),
        "w_out": jp.Dense(hidden, N_CLASSES, key=k4, gain=gain),
        "out": jp.LeakyIntegrator(tau=tau),
    },
    edges=[
        ("input", "w_in"), ("w_in", "h1"),
        ("h1", "w_rec"), ("w_rec", "h1"),        # the cycle
        ("h1", "w_h2"), ("w_h2", "h2"),
        ("h2", "w_out"), ("w_out", "out"),
    ],
    output="out",
)
```

Two things change with recurrence:

**The recurrent gain is 0.2, not 6.33.** Its output is summed into the same membrane on the next
step, so a gain sized for feedforward drive makes the loop self-amplifying and the network
saturates within a few timesteps.

**Parallel-in-time no longer applies.** A cycle in time cannot be solved by an associative scan,
so use `jp.unroll`. The `Graph` refuses `unroll_parallel` by name rather than silently computing
something else.

## 8. Augmentation

SHD has only 8156 training samples and these models reach 0.96+ train accuracy, so the ceiling
is memorization, not capacity. Both transforms in `examples/shd.py` are label-preserving:

```python
xs = jnp.roll(xs, offset, axis=0)                     # random time shift, +/- 20 steps
xs = xs * (jax.random.uniform(key, xs.shape) >= 0.1)  # input spike dropout
```

A spoken digit shifted a few milliseconds is the same digit, and dropping input spikes mimics
the variability of the cochlear model that generated them.

## Where the remaining gap is

0.696 against a ~0.71 reference, with 0.96 train accuracy. The limit is regularization rather
than architecture. Promising directions, none of them yet swept: dropout between layers, a
learning-rate schedule, and a search over the recurrent gain, which is set to 0.2 here.
