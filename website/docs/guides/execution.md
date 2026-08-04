---
id: execution
title: Execution and parallel-in-time
sidebar_position: 3
---

# Execution and parallel-in-time

Three ways to run a network over time. They have identical signatures, so switching is a
one-word change, and they trade against each other in ways worth understanding before you pick.

```python
spikes, state = jp.unroll(net, xs)               # sequential reference
spikes, state = jp.unroll_checkpointed(net, xs)  # same result, O(sqrt(T)) memory
spikes, state = jp.unroll_parallel(net, xs)      # associative scan over time
```

| | Works with | Time | Peak memory |
|---|---|---|---|
| `unroll` | everything | baseline | `O(T·B·N)` |
| `unroll_checkpointed` | everything | 1.1–1.4× slower | `O(sqrt(T)·B·N)` — **67× less at T=5000** |
| `unroll_parallel` | reset-free neurons only | **~2.5× faster end to end** | materializes `[T, B, N]` |

## `unroll` — the sequential reference

One `lax.scan` step per timestep, membrane state materialized at every step so BPTT can reach
it. This is the correctness reference for the whole library and the only path that supports
every neuron model.

State is explicit, so chunking a long sequence is exact rather than approximate:

```python
spikes_a, state = jp.unroll(net, xs[:50])
spikes_b, state = jp.unroll(net, xs[50:], state)   # exactly equals the unchunked run
```

That is also truncated BPTT: stop the gradient on `state` between chunks.

## `unroll_checkpointed` — rematerialization

Identical results to `unroll`, but the backward pass re-runs each chunk's forward instead of
keeping every timestep's residuals live. Peak scratch goes from `O(T·B·N)` to roughly
`O(sqrt(T)·B·N)` at the cost of one extra forward pass.

Measured on a T4, 4-layer network, batch 8:

| T | naive | checkpointed | reduction | time cost |
|---:|---:|---:|---:|---:|
| 100 | 5.4 MB | 910 KB | 6.0× | 1.29× |
| 1,000 | 52.8 MB | 2.0 MB | 26.4× | 1.15× |
| 5,000 | 263.8 MB | 3.9 MB | **67.0×** | 1.09× |

The saving grows with `T` because checkpointed memory scales as `sqrt(T)` while naive scales as
`T`. Rematerialization is *cheaper* on GPU than on CPU (1.09× versus about 1.4× at the long
end), because the extra forward pass uses compute the GPU has spare while the memory traffic it
avoids is the real bottleneck.

The chunk size defaults to the divisor of `T` closest to `sqrt(T)`, which is where the cost is
minimized. It is restricted to exact divisors on purpose: padded timesteps would still advance
the recurrence and silently corrupt the returned final state. Override it with `chunk_size=`,
and it must divide `T`.

## `unroll_parallel` — solving the whole time axis at once

A reset-free membrane is an affine recurrence, `v[t] = a·v[t-1] + b[t]`. Affine maps compose
associatively, so `jax.lax.associative_scan` solves all `T` steps with `O(log T)` depth instead
of `O(T)` sequential steps. Same arithmetic, different dependency structure.

**The results are bit-identical**, which is the thing to check first for a rewrite like this.
Membrane values differ by at most 5.96e-07 at `T=8192` — pure float32 accumulation-order noise
— and the binary spike trains match exactly at every length tested.

### The numbers, and which one to believe

| What was measured | Speedup |
|---|---|
| Isolated membrane, `T=8192` | 119× |
| Whole network forward+backward, `T=8192` | 17–22× |
| **A real SHD training epoch** | **~2.5×** |

All three measure different things, and the last is the one that describes a training run.
An epoch also contains host-to-device transfer, the Dense matmuls, the optimizer and
evaluation, none of which parallelizing the time axis touches; Amdahl's law does the rest.
Plan around 2.5×: the 119× is a microbenchmark of a single membrane and does not describe
end-to-end training.

### What it costs: memory

The parallel path materializes the full `[T, B, N]` activation tensor — 1.9 GB of scratch at
`T=8192` where the sequential checkpointed path needs a few megabytes. So parallel-in-time and
rematerialization currently sit at opposite ends of a tradeoff rather than composing. With a
long sequence and a small GPU, checkpointing is still the right choice. (Chunked parallel scan
— parallel within a chunk, checkpointed across chunks — would make them compose and is not
implemented yet.)

### What it works with

`unroll_parallel` requires every layer to be either stateless or reset-free:

- ✅ `LinearLIF`, `LeakyIntegrator`
- ✅ `Dense`, `Conv2d`, `Pool2d`, `Flatten` — stateless, applied per timestep
- ❌ `LIF`, `ALIF` — reset makes the recurrence nonlinear
- ❌ `Izhikevich` — nonlinear in time by construction
- ❌ any recurrent `Graph` — a cycle in time cannot be resolved by an associative scan

Unsupported layers raise by name rather than silently falling back, because a silent fallback
turns a correctness question into a performance mystery.

Reset-free neurons are a real published model class (the PSN / parallel spiking network
literature), not a workaround. The tradeoff is real too: without reset a neuron cannot regulate
its own firing, so a strongly driven unit saturates at one spike per timestep.

### Why reset cannot use this path

Reset makes the recurrence nonlinear, and this is not a matter of implementation effort — it
was attempted and measured. A chunked fixed-point scheme (guess spikes → solve the resulting
linear system in parallel → recompute spikes, to convergence) is provably exact, but the
iterations needed grow roughly linearly with chunk size: 5 iterations at chunk 8, 11 at 32, 33
at 128, 110 at 512. Every bit of parallelism a larger chunk buys is spent immediately on more
sequential passes. The best configuration measured **0.99× — a wash.**

Plain Jacobi iteration over the whole sequence oscillates: from "no resets" it produces too many
spikes, applying all those resets produces none, and it 2-cycles forever. Damping and a
sigmoid-annealing homotopy both collapse to "predict no spikes."

The full write-up is in `benchmarks/README.md` §6.

## Choosing

- **Long sequences, memory-bound, any neuron:** `unroll_checkpointed`.
- **Reset-free network, speed-bound, memory to spare:** `unroll_parallel`.
- **Anything else, or when debugging:** `unroll`. It is the reference the others are checked
  against.

## Readouts

All three return per-timestep outputs stacked on the leading axis. Reduce them with:

```python
jp.spike_rate(spikes)             # mean over time -- the rate-coded readout
jp.spike_count(spikes)            # sum over time
jp.max_membrane_logits(membrane)  # max over time; use with a LeakyIntegrator readout
jp.density(spikes)                # fraction of (timestep, neuron) slots that spiked
```

`density` is worth watching during development. It is how you catch a layer that has gone
silent, and below roughly 10% density a sparse gather-and-accumulate path would beat a dense
matmul — which is why the benchmarks report it.
