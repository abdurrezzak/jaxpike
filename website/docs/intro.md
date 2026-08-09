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

Every number here is reproducible from `benchmarks/` in the repository, on an NVIDIA T4. The
[benchmarks page](./benchmarks.md) records unfavourable results alongside the favourable ones,
including the cases where an approach did not pay off.

Training the same SHD network in every framework, side by side in one container on one GPU:

| framework | 20 epochs, batch 256, T=256 | peak memory |
|---|---:|---:|
| SpikingJelly, multi-step + CuPy | **6.02 s** | 792.1 MB |
| **jaxpike, `unroll`** | **8.12 s** | 324.5 MB |
| **jaxpike, `unroll_checkpointed`** | 11.07 s | **64.2 MB** |
| Norse | 252.21 s | 737.3 MB |
| snnTorch | 347.18 s | 675.8 MB |

Accuracy on SHD is **0.751**, matching the 0.70–0.75 band published for Spyx under the same
protocol. Other measured results:

| Result | Number |
|---|---|
| Parallel-in-time, isolated membrane | 119× faster at `T=8192` |
| BPTT memory via rematerialization | 67× less at `T=5000` |
| e-prop memory | flat in `T` — 2671× less than BPTT at `T=4000` |
| LIF integrator | exact closed-form ODE solution |

The 119× is an isolated membrane microbenchmark. A real training epoch also contains data
movement, matrix multiplies and the optimizer, none of which parallelizing the time axis
touches — which is why the end-to-end table above is the one to plan around.

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
