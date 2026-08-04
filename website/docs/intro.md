---
id: intro
title: jaxpike
sidebar_label: Introduction
slug: /
---

# jaxpike

Fast, flexible spiking neural networks in JAX.

:::warning Pre-alpha
The library works and the numbers below are measured, but the API is not stable yet and
jaxpike is not on PyPI. Install from a checkout.
:::

## What it is

A spiking neural network library built on JAX and [Equinox](https://github.com/patrick-kidger/equinox).
Networks are ordinary pytrees, state is explicit and functional, and everything composes with
`jax.jit`, `jax.grad` and `jax.vmap` the way any other JAX code does.

The field usually frames SNN tooling as a single tradeoff: **speed versus flexibility.**
Libraries built on hand-written CUDA kernels are fast but only support the neuron models
somebody already wrote a kernel for. Libraries in pure PyTorch or JAX let you write any neuron
and run considerably slower. jaxpike attacks that tradeoff from the algorithmic side first —
parallel-in-time execution, rematerialization and online learning are all pure JAX and need no
kernel code at all.

## What is measured

Every number here is reproducible from `benchmarks/` and `examples/` in the repository, on an
NVIDIA T4. The [benchmarks page](./benchmarks.md) records unfavourable results alongside the
favourable ones, including the cases where an approach did not pay off.

| Result | Number |
|---|---|
| Parallel-in-time, isolated membrane | 119× faster at `T=8192` |
| Parallel-in-time, whole network training | **~2.5× faster end to end** |
| BPTT memory via rematerialization | 67× less at `T=5000` |
| e-prop memory | flat in `T` — 2671× less than BPTT at `T=4000` |
| Spiking Heidelberg Digits, recurrent | **0.696 test** (published reference ~0.71) |
| LIF integrator | exact closed-form ODE solution |

The 2.5× is the figure to plan around. The 119× is an isolated membrane microbenchmark; a real
training epoch also contains data transfer, matmuls and the optimizer, none of which
parallelizing the time axis touches.

## Where to start

- **New to the library:** [Installation](./getting-started/installation.md) then the
  [quickstart](./getting-started/quickstart.md).
- **Coming from snnTorch:** read [the migration guide](./guides/coming-from-snntorch.md) first.
  Two conventions differ numerically and both will silently change your results.
- **Your network won't train:** [Why deep SNNs go silent](./guides/silent-networks.md). This is
  the most common real failure in the field.
- **Chasing speed:** [Execution and parallel-in-time](./guides/execution.md).

## Design decisions to know early

**Neurons are a contract, not a base class.** Any module with `init_state`, `out_shape` and
`__call__(state, x) -> (state, spikes)` works everywhere in the library. Nothing is registered
or subclassed.

**Surrogate gradients are smooth relaxations, differentiated by autodiff.** You write one
function; there is no custom VJP to get wrong, and the derivative can be finite-difference
tested.

**The LIF input is normalized by `(1 - alpha)`.** A constant drive `x` settles at exactly `x`,
so inputs are in threshold units. This is the convention snnTorch does not use, and it is the
main thing to know when porting weights.

**Membrane state is always float32**, even under bf16 training, because a leaky integrator runs
for thousands of steps and low-precision accumulation drifts enough to flip threshold crossings.
