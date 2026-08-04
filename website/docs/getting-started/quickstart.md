---
id: quickstart
title: Quickstart
sidebar_position: 2
---

# Quickstart

Everything on this page runs on CPU in a few seconds.

## A network

```python
import jax
import jax.numpy as jnp
import jaxpike as jp

key = jax.random.key(0)
k1, k2 = jax.random.split(key)

net = jp.Sequential(
    jp.Dense(784, 512, key=k1),
    jp.LIF(tau=20.0),                    # tau is a learnable parameter by default
    jp.Dense(512, 10, key=k2),
    jp.LIF(tau=20.0, surrogate=jp.ATan()),
)

xs = jax.random.uniform(key, (100, 32, 784))   # (time, batch, features)
spikes, final_state = jp.unroll(net, xs)
logits = jp.spike_rate(spikes)                 # rate-coded readout
```

Three conventions to absorb from that snippet:

**Time is the leading axis.** Inputs are `(time, batch, ...)`, and `unroll` scans over it.

**State is returned, never stored on the module.** `net` is an immutable pytree; the membrane
lives in `final_state`.

**`tau` is learnable.** It is stored as `log_tau` so it stays positive under unconstrained
optimization. Freeze it with `equinox.partition` if you don't want it trained.

## Chunking long sequences

Because state is explicit, truncated BPTT and streaming come for free and cost nothing to
express:

```python
spikes_a, state = jp.unroll(net, xs[:50])
spikes_b, state = jp.unroll(net, xs[50:], state)   # exactly equals the unchunked run
```

## Training step

```python
import equinox as eqx
import optax

optimizer = optax.adamw(2e-3)
opt_state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))

def loss_fn(model, xs, labels):
    spikes, _ = jp.unroll(model, xs)
    logits = jp.spike_count(spikes)
    return jp.cross_entropy(logits, labels), jp.accuracy(logits, labels)

step = jp.make_step(loss_fn, optimizer)          # jitted; returns (model, opt_state, loss, aux)

labels = jnp.zeros((32,), dtype=jnp.int32)
net, opt_state, loss, acc = step(net, opt_state, xs, labels)
```

`make_step` expects `loss_fn` to return `(loss, aux)` and handles the `eqx.filter_value_and_grad`
plumbing.

For classification, prefer a `LeakyIntegrator` readout over counting spikes. Counting means the
loss only sees an output unit once it crosses threshold, so a class that never fires produces no
gradient and can never learn to fire:

```python
net = jp.Sequential(
    jp.Dense(784, 512, key=k1),
    jp.LinearLIF(tau=20.0, threshold=0.5),
    jp.Dense(512, 10, key=k2),
    jp.LeakyIntegrator(tau=20.0),     # outputs membrane, never spikes
)
membrane, _ = jp.unroll(net, xs)
logits = jp.max_membrane_logits(membrane)
```

## Making it fast

Swap the runner. `unroll_parallel` solves the whole time axis with an associative scan instead
of stepping through it, which works for reset-free neurons (`LinearLIF`, `LeakyIntegrator`,
and any stateless layer):

```python
membrane, _ = jp.unroll_parallel(net, xs)     # ~2.5x faster end to end on a real epoch
```

`unroll_checkpointed` is the other axis — same results as `unroll`, `O(sqrt(T))` memory instead
of `O(T)`:

```python
membrane, _ = jp.unroll_checkpointed(net, xs)
```

Both are covered in [Execution](../guides/execution.md), including when each one does not apply
and what it does instead of failing silently.

## Defining a neuron

Nothing is registered or subclassed. Any module with these three methods works everywhere in
the library, including in `Sequential`, `Graph` and the visualization functions:

```python
init_state(input_shape) -> state pytree
out_shape(input_shape)  -> output shape
__call__(state, x)      -> (new_state, spikes)
```

## Defining a surrogate gradient

Write the smooth relaxation of the Heaviside step; autodiff supplies the derivative:

```python
class MySurrogate(jp.Surrogate):
    slope: float = 10.0

    def relaxation(self, v):
        return jax.nn.sigmoid(self.slope * v)
```

The forward pass still emits exact binary spikes — `Surrogate.__call__` applies the
straight-through identity `soft + stop_gradient(hard - soft)`.

## Next

- [Train SHD end to end](./training-shd.md)
- [Why deep SNNs go silent](../guides/silent-networks.md) — read this before scaling depth
