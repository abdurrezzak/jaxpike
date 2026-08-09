---
id: benchmarks
title: Benchmarks
sidebar_position: 99
---

# Benchmarks

Benchmarks are published with their unfavourable results intact, including approaches that were
measured and abandoned. Knowing which techniques do *not* pay off is as useful when choosing an
execution path as knowing which do.

All GPU numbers are an **NVIDIA T4**, the weakest GPU available on Modal's free tier, and should
improve on better hardware. The full write-ups are in `benchmarks/README.md`.

```bash
PYTHONPATH=. python benchmarks/memory_scaling.py [--time]
PYTHONPATH=. python benchmarks/parallel_scan.py
python -m modal run benchmarks/gpu/run_modal.py --bench {memory,parallel,network,reset,shd}
python -m modal run benchmarks/gpu/run_torch_comparison.py --suite speed
```

## Against other frameworks

Every framework installed side by side and run in one container on one GPU, each in its own
subprocess so no library is measured while another holds device memory. Identical model,
identical input arrays, same optimizer, loss and dtype:
`Linear(128,128) -> LIF -> Linear(128,128) -> LIF -> Linear(128,20) -> LI`, Adam 5e-4,
cross-entropy on the time-integrated readout with 0.3 label smoothing, fp32, 20 epochs at
batch 256, T=256.

| framework | training time | peak memory |
|---|---:|---:|
| SpikingJelly 0.0.0.0.14, multi-step + CuPy | **6.02 s** | 792.1 MB |
| **jaxpike, `unroll`** | **8.12 s** | 324.5 MB |
| **jaxpike, `unroll_checkpointed`** | 11.07 s | **64.2 MB** |
| jaxpike, `unroll_parallel` | 15.80 s | 288.1 MB |
| Norse 1.1.0 | 252.21 s | 737.3 MB |
| SpikingJelly, Torch backend | 260.62 s | 696.3 MB |
| snnTorch 1.0.0 | 347.18 s | 675.8 MB |

jaxpike figures are steady state; compilation costs a further 14–19 s, paid once per shape.
Test accuracy is 0.751, against the 0.70–0.75 band published for Spyx under this protocol.

Three concessions are made to the other frameworks deliberately, so that a favourable result
cannot be dismissed: their leaky readout is evaluated in closed form rather than looped,
SpikingJelly runs its fastest documented path with the CuPy backend verified at runtime rather
than assumed, and each framework keeps its native decay parameterization.

Anything that steps through time in a Python loop is 31–43× slower. SpikingJelly's fused CuPy
kernel is **1.35× faster than the best jaxpike path** and is not beaten. Ablation locates the
whole gap: 83% of a jaxpike training step is the neuron time loop and 71% is the backward
pass, while the hoisted `Dense` layers already compile to a single large matrix multiply.

## How memory is measured

`jax.jit(fn).lower(...).compile().memory_analysis().temp_size_in_bytes` — XLA's own peak scratch
allocation for the compiled executable. This matters because the SNN benchmarking literature
reports speed and stays quiet about memory, which is where the largest differences actually are.

It is deterministic, has no allocator noise, and works on a CPU-only laptop. Two limits:
it is XLA's *plan* rather than an observed high-water mark, and it excludes arguments and
outputs. It is the right instrument for comparing how two implementations scale against each
other, and the wrong one for predicting whether a model fits on a given GPU. The CPU and GPU
measurements agree to within 0.6%, which is what makes the laptop numbers trustworthy.

## Memory: rematerialization

Forward+backward through a 4-layer network, batch 8, on a T4.

| T | naive | checkpointed | reduction | time cost |
|---:|---:|---:|---:|---:|
| 100 | 5.4 MB | 910 KB | 6.0× | 1.29× |
| 500 | 26.5 MB | 1.6 MB | 16.6× | 1.00× |
| 1,000 | 52.8 MB | 2.0 MB | 26.4× | 1.15× |
| 2,500 | 131.9 MB | 3.5 MB | 37.2× | 1.10× |
| 5,000 | 263.8 MB | 3.9 MB | **67.0×** | 1.09× |

Naive memory is almost perfectly linear in `T` — bytes per step vary by only 1.03× across a 50×
range of sequence lengths — which confirms the `O(T·B·N)` claim. The saving *grows* with `T`
because checkpointed memory scales as `sqrt(T)`.

A practical consequence: pure-JAX `jax.checkpoint` captures most of the available memory win in
about fifteen lines, so hand-written fused kernels would have to justify themselves on
bandwidth and speed rather than memory, and be benchmarked against **this** path rather than
the naive scan.

## Parallel-in-time

Isolated reset-free membrane, batch 8, 128 neurons:

| T | sequential | associative | speedup | spikes |
|---:|---:|---:|---:|:---:|
| 128 | 0.0018 s | 0.0004 s | 4.35× | exact match |
| 512 | 0.0068 s | 0.0002 s | 33.62× | exact match |
| 2,048 | 0.0255 s | 0.0003 s | 83.68× | exact match |
| 8,192 | 0.1032 s | 0.0009 s | **119.10×** | exact match |

Whole network forward+backward (Dense 128→512 → LinearLIF → Dense 512→512 → LinearLIF, batch 16):

| T | train seq | train par | speedup | parallel scratch |
|---:|---:|---:|---:|---:|
| 128 | 0.0114 s | 0.0012 s | 9.4× | 25.8 MB |
| 2,048 | 0.1365 s | 0.0079 s | 17.3× | 489.0 MB |
| 8,192 | 0.5415 s | 0.0249 s | **21.7×** | 1.9 GB |

A real SHD training epoch, wall clock including transfer, optimizer and evaluation:

| timesteps | batch | sequential | parallel | speedup |
|---:|---:|---:|---:|---:|
| 250 | 128 | 3.9 s | 1.5 s | 2.6× |
| 1,000 | 64 | 23.7 s | 10.0 s | 2.4× |

**These three numbers measure different things and only the last describes a training run.**
The 119× is an isolated membrane and the 17–22× is an isolated forward/backward pass; an epoch
also contains host-to-device transfer, matrix multiplies, the optimizer and evaluation, none of
which parallelizing time touches. On the framework comparison above, at `T=256` with batch 256,
`unroll_parallel` is *slower* than `unroll` — the time axis is not the bottleneck at that
shape.

The parallel path also costs memory — it materializes the full `[T, B, N]` tensor, 1.9 GB at
`T=8192` where checkpointing needs a few megabytes. The two techniques currently sit at
opposite ends of a tradeoff rather than composing.

On CPU the associative scan is 3–5× *slower*, and that is expected: it performs roughly 2× the
total work in exchange for `O(log T)` depth instead of `O(T)`. That is a losing trade on a
handful of cores and a winning one on thousands of lanes.

## Negative result: reset-based parallel-in-time

Reset makes the recurrence nonlinear, so the associative scan does not apply. The candidate
tested was a chunked fixed point — sequential across chunks with an exact carry, and within each
chunk iterate *guess spikes → solve the resulting linear system in parallel → recompute spikes*
until the spike train stops changing. The fixed point is provably the exact sequential answer.

T=2048, batch 16, 512 neurons, T4. Sequential scan: 0.0216 s.

| chunk | iterations/chunk | wrong spikes | time | vs sequential |
|---:|---:|---:|---:|---:|
| 8 | 5.0 | 0 / 16.7M | 0.0242 s | 0.92× |
| 32 | 11.0 | 4 / 16.7M | 0.0224 s | **0.99×** |
| 128 | 32.6 | 2 / 16.7M | 0.0330 s | 0.68× |
| 512 | 109.5 | 2 / 16.7M | 0.1152 s | 0.19× |

**It does not pay off.** The best configuration is 0.99× — a wash. The reason is visible in the
table: iterations grow roughly linearly with chunk size, so every bit of parallelism a larger
chunk buys is spent immediately on more sequential passes. There is no window where the trade
wins.

Two simpler schemes were tried first and fail outright. Plain Jacobi iteration over the whole
sequence oscillates: from "no resets" it produces too many spikes, applying all those resets
produces none, and it 2-cycles forever. Damping (λ = 0.3/0.5/0.7) and a sigmoid-annealing
homotopy both converge to "predict no spikes."

Parallel-in-time is therefore a reset-free feature. The schemes above cover the obvious
approaches to the reset case; none of them wins.

The handful of wrong spikes are float32 noise rather than a defect: 4 in 16.7 million, all at
threshold crossings where two summation orders straddle the boundary by ~1e-7. This generalizes
— **bit-exact spike comparison is fragile near threshold** whenever reduction order changes, so
compare with a tolerance or compare rates.

## Spiking Heidelberg Digits

Network: `Dense(700,256) → LinearLIF → Dense(256,256) → LinearLIF → Dense(256,20) →
LeakyIntegrator`, max-membrane readout, AdamW.

| | validation | **test** |
|---|---:|---:|
| jaxpike, feedforward + augmentation | 0.888 | **0.626** |
| jaxpike, **recurrent** + augmentation | 0.839 | **0.696** |
| Cramer et al. 2020, feedforward reference | — | ~0.48 |
| Cramer et al. 2020, recurrent reference | — | ~0.71 |

Feedforward comfortably beats its published baseline; recurrent lands just under.

**Protocol note, because it changes the numbers.** These runs hold out 10% of train as
validation, pick the epoch by validation accuracy, and report test once at that epoch.
Reporting the best test accuracy across epochs instead selects the epoch *on the test set*,
which inflates the figure by 0.5 to 4 points depending on the run.

The protocol is also what makes the next result visible. Recurrence is worth +7.0 points on
test while scoring *lower* on validation (0.839 against 0.888). Validation is drawn from the
training pool, but SHD's test set holds out entire speakers — so recurrence buys speaker
generalization specifically, rather than raw capacity. A test-selected protocol never looks at
validation and so hides the distinction entirely.

## e-prop

Peak scratch bytes for one gradient, 2-layer network:

| T | BPTT | e-prop | ratio |
|---:|---:|---:|---:|
| 100 | 210,984 | 3,128 | 68× |
| 500 | 1,046,184 | 3,128 | 335× |
| 1,000 | 2,090,024 | 3,128 | 668× |
| 4,000 | 8,354,024 | 3,128 | **2671×** |

The e-prop column is identical at every length. Gradient alignment against BPTT, and the two
sources of approximation, are covered in [Online learning](./guides/online-learning.md).

## Two pitfalls these runs surfaced

Both are easy to hit when benchmarking any SNN, in this library or another, and both distort
results rather than failing outright.

**Keep the dataset off the accelerator.** A loader returning device arrays pins the whole
dataset in GPU memory: SHD at 1000 timesteps needs 22.8 GB in float32, which exhausts a 16 GB
card before training starts. Transfer one batch at a time.

**Store binary spikes as uint8, not float32.** Doing so cuts SHD from 22.8 GB to 5.3 GB and
per-batch PCIe traffic by 4×. With float32 storage the T=1000 comparison measured 1.56× where
the true figure was 2.4×, because the input pipeline had become the bottleneck and was masking
the compute win.

## What is not benchmarked yet

Fused Pallas kernels. Two findings above set a high bar for them: `jax.checkpoint` already
captures most of the memory win, and fusing does not save `T` kernel launches, because
`lax.scan` already lowers to a single XLA while-loop. The remaining wins are HBM traffic and
residual storage, so the bar for adopting a kernel path is 2× over `unroll_checkpointed`
rather than over the naive scan.
