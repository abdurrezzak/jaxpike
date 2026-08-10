---
id: long-sequences
title: Training on long sequences
sidebar_position: 4
---

# Training on long sequences

Backpropagation through time stores every timestep's intermediates so the backward pass can
reach them. That is `O(T · batch · N)` memory, linear in sequence length with no upper bound,
and it is the reason spiking networks are usually trained on a few hundred timesteps rather
than a few thousand.

jaxpike offers three ways to run a network over time. They accept the same arguments, return
the same things, and agree numerically. This tutorial covers how to choose between them.

## The three strategies

```python
spikes, state = jp.unroll(net, xs)                # sequential BPTT
spikes, state = jp.unroll_checkpointed(net, xs)   # O(sqrt(T)) memory
spikes, state = jp.unroll_parallel(net, xs)       # O(log T) depth, reset-free neurons only
```

| | works with | time | peak memory |
|---|---|---|---|
| `unroll` | every neuron | baseline | `O(T · B · N)` |
| `unroll_checkpointed` | every neuron | ~1.1–1.4× slower | `O(sqrt(T) · B · N)` |
| `unroll_parallel` | reset-free neurons only | faster only at long `T` or small batch | materializes `[T, B, N]` |

## Watching memory grow

The cost is easy to see directly. XLA reports the peak scratch it plans to allocate, which is
deterministic and works without a GPU:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxpike as jp

k1, k2 = jax.random.split(jax.random.key(0))
net = jp.Sequential(
    jp.Dense(64, 128, key=k1), jp.LIF(tau=20.0),
    jp.Dense(128, 10, key=k2), jp.LIF(tau=20.0),
)


def peak_scratch(runner, xs):
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def loss(p):
        return jnp.sum(runner(eqx.combine(p, static), xs)[0])

    compiled = jax.jit(jax.grad(loss)).lower(params).compile()
    return compiled.memory_analysis().temp_size_in_bytes / 2**20


for steps in (100, 1000, 5000):
    xs = jnp.zeros((steps, 8, 64))
    naive = peak_scratch(jp.unroll, xs)
    remat = peak_scratch(jp.unroll_checkpointed, xs)
    print(f"T={steps:<5} unroll {naive:7.1f} MB   checkpointed {remat:5.1f} MB   {naive/remat:.0f}x")
```

```
T=100   unroll     5.4 MB   checkpointed   0.5 MB   10x
T=1000  unroll    35.8 MB   checkpointed   1.6 MB   22x
T=5000  unroll   178.9 MB   checkpointed   3.2 MB   56x
```

Two things to read off this. `unroll` is linear in `T`, as the `O(T · B · N)` bound says it
should be. And the saving from checkpointing **grows** with sequence length, because
checkpointed memory scales as `sqrt(T)` — so the longer your sequence, the more the trade is
worth taking.

## Choosing

**Start with `unroll`.** It is the reference implementation, it supports every neuron, and at a
few hundred timesteps memory is rarely the constraint.

**Switch to `unroll_checkpointed` when you run out of memory.** It is exact, works with every
neuron including reset, and costs roughly 10–40% more time for the extra forward pass. On the
SHD benchmark it holds a 256-step BPTT graph in 64 MB where the unremat'd path needs 325 MB.
This is the path that turns "does not fit" into "fits".

**Reach for `unroll_parallel` only after measuring.** It removes the sequential dependency in
time by solving a linear recurrence with an associative scan, which is `O(log T)` depth instead
of `O(T)`. That is a large win when the time axis is the bottleneck — 119× on an isolated
membrane at `T=8192` — and no win at all when it is not. On the SHD benchmark at `T=256` with
batch 256 it is *slower* than `unroll`, because there is already enough parallel work to fill
the device.

It also requires **reset-free neurons**. Reset makes the recurrence nonlinear, so `LIF` does not
qualify and `LinearLIF` does. A layer without a `parallel_apply` is named in the error rather
than silently falling back:

```python
jp.unroll_parallel(jp.Sequential(jp.LIF(tau=20.0)), xs)
# TypeError: layer 0 (LIF) has no parallel_apply, so this network cannot run
# parallel-in-time. Reset-based neurons are nonlinear in time; swap in LinearLIF or use
# unroll()/unroll_checkpointed().
```

Reset is not free to give up: on SHD it is worth about 14 accuracy points, which is usually a
worse trade than the time it saves.

## Streaming a sequence too long to hold

State is explicit and functional, so a sequence that does not fit in memory at all can be
processed in chunks by feeding the returned state back in. The result is exactly the unchunked
run — not an approximation:

```python
xs = jnp.zeros((1000, 8, 64))

whole, _ = jp.unroll(net, xs)

first, state = jp.unroll(net, xs[:500])
second, _ = jp.unroll(net, xs[500:], state)

assert jnp.allclose(whole, jnp.concatenate([first, second]))
```

Truncating the gradient between chunks — truncated BPTT — comes from the same mechanism: wrap
the carried state in `jax.lax.stop_gradient` before feeding it forward, and each chunk's
backward pass stops at its own boundary.

## Tuning the time loop

Both sequential paths take `scan_unroll`, which controls how many timesteps are emitted per
loop iteration. A neuron step is a handful of elementwise operations — too little work to hide
a kernel launch — so emitting several per iteration lets the compiler fuse them.

```python
spikes, state = jp.unroll(net, xs, scan_unroll=8)   # the default
```

The default of 8 was chosen by measurement on the SHD training step, where it is 2.25× faster
than one step per iteration. Larger values stop paying and grow compile time sharply. Note that
changing it perturbs results at the level of float reassociation, around 1e-9 — small, but not
zero.

`unroll` also takes `remat_step=True`, which recomputes the neuron step in the backward pass
instead of storing its residuals: about 22% less peak scratch for about 9% more time, with
bit-identical gradients.

## See also

- [Execution guide](../guides/execution.md) — the measurements behind these recommendations
- [Benchmarks](../benchmarks.md) — full protocol, including where each path loses
