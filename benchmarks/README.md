# Benchmarks

Results are committed as they are measured, including unfavourable ones. A benchmark suite
that only records wins is marketing.

## Running

```bash
PYTHONPATH=. python benchmarks/memory_scaling.py [--time]
PYTHONPATH=. python benchmarks/parallel_scan.py
```

## How memory is measured

`jax.jit(fn).lower(...).compile().memory_analysis().temp_size_in_bytes` — XLA's own peak
scratch allocation for the compiled executable. This matters because the SNN benchmarking
literature reports speed and stays quiet about memory, which is where the largest differences
actually are; Spyx's paper says outright that it could not report memory because JAX has no
`torch.cuda.max_memory_allocated` equivalent.

It is deterministic, has no allocator noise, and works on a CPU-only laptop. Two honest
limits: it is XLA's *plan* rather than an observed high-water mark, and it excludes arguments
and outputs. It is the right instrument for comparing how two implementations scale against
each other, and the wrong one for predicting whether a model fits on a given GPU. Phase 0b
cross-checks it against real device measurements.

---

## Phase 0a results — 2026-08-01, Apple Silicon CPU, JAX 0.11.0

Both Phase 0a gates **pass**. These retire the plan's largest risk without renting a GPU.

### 1. BPTT memory scaling

Forward+backward through a 4-layer network (Dense→LIF→Dense→LIF), batch 8, 64 features,
128 hidden.

| T | naive scan | checkpointed | reduction | naive bytes/step |
|---:|---:|---:|---:|---:|
| 100 | 5.4 MB | 937.0 KB | 5.9× | 56,926 |
| 500 | 26.5 MB | 1.6 MB | 16.4× | 55,622 |
| 1,000 | 52.9 MB | 2.0 MB | 26.1× | 55,459 |
| 2,500 | 132.0 MB | 3.6 MB | 37.0× | 55,361 |
| 5,000 | 263.8 MB | 4.0 MB | **66.6×** | 55,329 |

**Gate 0a-1 (≥5× at T=5000): PASS at 66.6×.**

Two things worth reading off this table. Naive memory is almost perfectly linear in `T` —
bytes-per-step varies by only 1.03× across a 50× range of sequence lengths — which confirms
the `O(T·B·N)` claim the whole thesis rests on. And the saving *grows* with `T`, because
checkpointed memory scales as `√T` while naive scales as `T`.

Time cost of rematerialization, same configuration:

| T | naive | checkpointed | slowdown |
|---:|---:|---:|---:|
| 100 | 0.0090 s | 0.0111 s | 1.24× |
| 500 | 0.0345 s | 0.0459 s | 1.33× |
| 1,000 | 0.0698 s | 0.0824 s | 1.18× |
| 2,500 | 0.1730 s | 0.2423 s | 1.40× |

1.2–1.4× time for 37× memory, which matches theory: checkpointing adds one extra forward
pass to a forward+backward that already costs about three, so ~1.33× is the expected price.

**Consequence for the plan.** The memory argument for writing Pallas kernels is now
*weaker*, not stronger, and that is a genuinely useful thing to learn this early: pure-JAX
rematerialization already captures most of the available memory win. Phase 2's fused kernels
must therefore justify themselves on **bandwidth and speed**, and must be benchmarked against
the checkpointed path rather than the naive one. Gate 0b has been rewritten accordingly.

### 2. Parallel-in-time (reset-free membrane)

Associative scan versus sequential scan, batch 8, 128 neurons.

| T | max membrane deviation | spike agreement | seq | par | speedup |
|---:|---:|:---:|---:|---:|---:|
| 128 | 3.58e-07 | exact match | 0.0001 s | 0.0003 s | 0.38× |
| 512 | 4.17e-07 | exact match | 0.0003 s | 0.0009 s | 0.29× |
| 2,048 | 4.77e-07 | exact match | 0.0008 s | 0.0032 s | 0.26× |
| 8,192 | 5.96e-07 | exact match | 0.0036 s | 0.0182 s | 0.20× |

No FLOP counts are reported here. XLA's cost analysis does not descend into while-loop
bodies, so it attributes a few thousand FLOPs to a sequential scan that performs millions —
a number wrong by three orders of magnitude is worse than no number.

**Gate 0a-2 (numerical equivalence): PASS.** Worst deviation 5.96e-07 at T=8192, pure float32
accumulation-order noise, and — the measure that actually matters — the binary spike trains
are *identical* at every length, so no threshold crossing was flipped.

**The associative scan is 3–5× slower here, and that is expected.** It performs roughly 2×
the total work in exchange for `O(log T)` depth instead of `O(T)`: a losing trade on a CPU
with a handful of cores, a winning one on hardware with thousands of lanes. Correctness is
what this run establishes. For the speedup, see the GPU results below.

---

## Phase 0b results — 2026-08-03, NVIDIA T4 (Modal), JAX 0.11.0

A T4 is the weakest GPU Modal offers. Everything here should improve on an A100 or H100.

### 3. Memory scaling reproduces on GPU

| T | naive | checkpointed | reduction | ckpt slowdown |
|---:|---:|---:|---:|---:|
| 100 | 5.4 MB | 910.3 KB | 6.0× | 1.29× |
| 500 | 26.5 MB | 1.6 MB | 16.6× | 1.00× |
| 1,000 | 52.8 MB | 2.0 MB | 26.4× | 1.15× |
| 2,500 | 131.9 MB | 3.5 MB | 37.2× | 1.10× |
| 5,000 | 263.8 MB | 3.9 MB | **67.0×** | 1.09× |

67.0× on GPU against 66.6× measured on CPU, which confirms `temp_size_in_bytes` is genuinely
device-independent and that the local measurements can be trusted. Rematerialization is also
*cheaper* on GPU than on CPU — 1.09× versus 1.4× at the long end — because the extra forward
pass is compute the GPU has spare while the memory traffic it avoids is the actual bottleneck.

### 4. Parallel-in-time, on hardware that can exploit it

Reset-free membrane, batch 8, 128 neurons.

| T | sequential | associative | speedup | spikes |
|---:|---:|---:|---:|:---:|
| 128 | 0.0018 s | 0.0004 s | 4.35× | exact match |
| 512 | 0.0068 s | 0.0002 s | 33.62× | exact match |
| 2,048 | 0.0255 s | 0.0003 s | 83.68× | exact match |
| 8,192 | 0.1032 s | 0.0009 s | **119.10×** | exact match |

**119× at T=8192, on the slowest GPU available, with bit-identical spike trains.** The
sequential scan's cost grows linearly in `T` while the associative scan's is nearly flat, so
the advantage widens with sequence length — exactly the asymptotic signature the approach
predicts, which is stronger evidence than any single number.

**This reorders the roadmap.** Parallel-in-time was listed third among Phase 2's work items,
behind fused kernels. It is now first: a 119× algorithmic win on a T4 dwarfs the 2× the
fused-kernel path is gated at, and it needs no Pallas, no custom VJP, and no hand-tuning.

**The limit, stated plainly.** This is the *reset-free* case, where the recurrence is linear.
Reset makes it nonlinear and this exact method no longer applies — that is what chunked scan
and the DEER-style tier of the plan are for, and neither is proven yet. Reset-free PSN-style
neurons are a real published model class, so the result is directly usable, but it is not yet
a general LIF result and must not be reported as one.


---

## Phase 2 results — 2026-08-03, NVIDIA T4 (Modal)

### 5. End-to-end network, parallel-in-time

What a user actually gets, as opposed to a microbenchmark: a full feedforward SNN
(Dense 128→512 → LinearLIF → Dense 512→512 → LinearLIF), batch 16, spike density ~35%.

| T | fwd seq | fwd par | fwd speedup | train seq | train par | **train speedup** | par scratch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.0029 s | 0.0006 s | 5.0× | 0.0114 s | 0.0012 s | **9.4×** | 25.8 MB |
| 512 | 0.0102 s | 0.0010 s | 10.1× | 0.0445 s | 0.0025 s | **17.7×** | 104.6 MB |
| 2,048 | 0.0311 s | 0.0021 s | 14.8× | 0.1365 s | 0.0079 s | **17.3×** | 489.0 MB |
| 8,192 | 0.1213 s | 0.0070 s | 17.4× | 0.5415 s | 0.0249 s | **21.7×** | 1.9 GB |

**21.7× faster training at T=8192**, on the weakest GPU available.

This is much less than the 119× measured on an isolated membrane, and the gap is the honest
part of the result: once the Dense layers are in the network they dominate the FLOP budget,
and they were never the sequential bottleneck. Parallelizing time removes the dependency
chain; it does not make the matmuls cheaper. 17–22× is the realistic figure to quote, and the
119× number should never be quoted as an end-to-end speedup.

**The cost, which is real: memory.** The parallel path materializes the full `[T, B, N]`
activation tensor — 1.9 GB of scratch at T=8192, where the sequential+checkpointed path needs
a few MB. So the two techniques currently sit at opposite ends of a tradeoff rather than
composing, and a user with a long sequence and a small GPU still wants checkpointing. Making
them compose (chunked parallel scan: parallel within a chunk, checkpointed across chunks) is
the obvious next piece of work and is not yet done.


### 6. Reset-based LIF, parallel-in-time — **negative result**

Reset makes the recurrence nonlinear, so the associative scan does not apply. The candidate
tested was a chunked fixed point: sequential across chunks with an exact carry, and within
each chunk iterate *guess spikes → solve the resulting linear system in parallel → recompute
spikes* until the spike train stops changing. The fixed point is provably the exact solution
(at t=0 the membrane depends on no spike, so s₀ is forced; by induction every later step is
forced too, making the fixed point unique and equal to the sequential answer).

T=2048, batch 16, 512 neurons, density 0.090, T4. Sequential scan: 0.0216 s.

| chunk | iterations/chunk | wrong spikes | time | vs sequential |
|---:|---:|---:|---:|---:|
| 8 | 5.0 (max 6) | 0 / 16.7M | 0.0242 s | 0.92× |
| 32 | 11.0 (max 13) | 4 / 16.7M | 0.0224 s | **0.99×** |
| 128 | 32.6 (max 33) | 2 / 16.7M | 0.0330 s | 0.68× |
| 512 | 109.5 (max 111) | 2 / 16.7M | 0.1152 s | 0.19× |

**It does not pay off.** The best configuration is 0.99× — a wash — and it degrades from
there. The reason is visible in the table: iterations needed grow roughly linearly with chunk
size (5 → 11 → 33 → 110 as chunk goes 8 → 32 → 128 → 512), so every bit of extra parallelism
bought by a larger chunk is spent immediately on more sequential iterations. Small chunks
converge fast but expose little parallelism; large chunks expose parallelism but need too
many passes. There is no window where the trade wins.

The handful of wrong spikes are float32 noise, not a bug: 4 in 16.7 million, all at threshold
crossings where two different summation orders straddle the boundary by ~1e-7. Worth
remembering as a general caution — bit-exact spike comparisons are fragile near threshold
whenever the reduction order changes.

**What was ruled out.** Two simpler schemes were tried first and both fail outright. Plain
Jacobi iteration over the whole sequence *oscillates*: starting from "no resets" produces too
many spikes, applying all those resets produces none, and it 2-cycles forever. Damping
(λ = 0.3/0.5/0.7) and a sigmoid-annealing homotopy both converge to "predict no spikes."

**Conclusion.** Parallel-in-time stays a reset-free feature. This is the risk the plan's
register flagged as most likely to fire, and it fired. The honest product position: reset-free
`LinearLIF` gets 21.7× and is a legitimate published model class (PSN); reset-based `LIF`
gets the sequential path with 67× memory reduction from checkpointing. Not solving reset is a
real limitation and should be stated as one rather than hidden.


---

## Real task: Spiking Heidelberg Digits — 2026-08-03, NVIDIA T4

SHD is the field's standard temporal benchmark: 20 spoken digits (English and German) turned
into spike trains over 700 cochlear channels. Unlike rate-coded MNIST the temporal structure
is genuine. Network: `Dense(700,256) → LinearLIF → Dense(256,256) → LinearLIF →
Dense(256,20) → LeakyIntegrator`, max-membrane readout, AdamW.

### Accuracy

**Protocol note, because it changes the numbers.** An earlier version of these results
reported the best test accuracy across epochs. That selects the epoch *on the test set* and
inflates the figure — measured here by about 0.5 to 4 points depending on the run. The
numbers below hold out 10% of train as validation, pick the epoch by validation accuracy, and
report test once at that epoch. Slightly worse numbers, and the only ones that will reproduce.

| | validation | **test** |
|---|---:|---:|
| jaxpike, feedforward + augmentation | 0.888 | **0.626** |
| jaxpike, **recurrent** + augmentation | 0.839 | **0.696** |
| Cramer et al. 2020, feedforward reference | — | ~0.48 |
| Cramer et al. 2020, recurrent reference | — | ~0.71 |

Feedforward comfortably beats its published baseline; recurrent lands just under the
published recurrent one.

**Recurrence is worth +7.0 points, and the reason is visible in the validation column.** The
feedforward model scores *higher* on validation (0.888 vs 0.839) and *lower* on test (0.626 vs
0.696). Validation is drawn from the training pool, but SHD's test set contains speakers held
out entirely — so a model can fit the training speakers well and still fail to generalize to
new ones. Recurrence buys speaker generalization specifically, not raw capacity. The
test-selected protocol hid this completely, because it never looked at validation at all.

Augmentation is a random time shift plus input spike dropout, both label-preserving: a spoken
digit shifted a few milliseconds is the same digit.

### Training speed, sequential versus parallel-in-time

Whole-epoch wall clock, including data transfer, optimizer and evaluation.

| timesteps | batch | sequential | parallel | speedup |
|---:|---:|---:|---:|---:|
| 250 | 128 | 3.9 s | 1.5 s | 2.6× |
| 1,000 | 64 | 23.7 s | 10.0 s | 2.4× |

**Calibration that matters: ~2.5× end-to-end, not the 17–21× from the microbenchmark.**
An epoch is more than time-stepping — host-to-device transfer, the Dense matmuls, the
optimizer and evaluation are all untouched by parallelizing the time axis, and Amdahl's law
does the rest. The 17–21× figure is the isolated forward/backward speedup and should always
be labelled as such. **~2.5× on real training is the number to quote to users.**

### Two bugs this run exposed, both worth keeping

**The whole dataset was being pinned in GPU memory.** The loader returned device arrays, so
SHD at 1000 timesteps tried to allocate 22.8 GB on a 16 GB card and OOMed before training
started. `iterate_batches` now requires host arrays and transfers only the current batch.

**Binary spikes were stored as float32.** Fixing that to uint8 cut the dataset from 22.8 GB
to 5.3 GB and, more importantly, cut per-batch PCIe traffic 4×. Before the fix the T=1000
comparison showed only 1.56× — the input pipeline had become the bottleneck and was masking
the compute win. After it, 2.4×. Worth remembering as a general rule: spike data is binary,
so storing or transferring it as float is always four times more traffic than necessary.


---

## Online learning (e-prop) — 2026-08-04

BPTT stores every timestep and walks backwards. e-prop carries an eligibility trace forward
and accumulates the weight gradient in place, so memory does not grow with sequence length.

### Memory: flat in T

Peak scratch bytes for one gradient, 2-layer network, measured through XLA's allocation plan.

| T | BPTT | e-prop | ratio |
|---:|---:|---:|---:|
| 100 | 210,984 | 3,128 | 68× |
| 500 | 1,046,184 | 3,128 | 335× |
| 1,000 | 2,090,024 | 3,128 | 668× |
| 4,000 | 8,354,024 | 3,128 | **2671×** |

The e-prop column is *identical* at every length. That is the whole point of the method, and
it is what makes training on arbitrarily long streams possible.

### Accuracy of the gradient

Cosine similarity against the true BPTT gradient, 40 timesteps.

| | reset-free (`LinearLIF`) | with reset (`LIF`) |
|---|---:|---:|
| layer feeding the loss | **1.000000** (exact to 2e-07) | 0.9988 |
| hidden layer | 0.879 | 0.917 |

Two independent sources of approximation, and it is worth keeping them apart. **Reset** feeds
a spike back into its own membrane, a temporal path the factorization does not carry; without
it, the membrane filter is the only route through time and the gradient is exact to float
precision. **Depth** is the other: a hidden layer's learning signal would have to be filtered
backwards through each membrane to be exact, which is a backward pass in time and the thing
online learning exists to avoid, so it is propagated spatially only. Cosine near 0.9 is a
well-aligned descent direction rather than the true gradient, and that is what makes it work.

### One bug worth recording

The first implementation evaluated the surrogate derivative at the **post-reset** membrane.
Since the threshold comparison happens before reset, this silently mis-placed the derivative
and dropped `LIF` gradient alignment to cosine 0.32 — while leaving `LinearLIF` exact, because
without reset the two membranes are the same value. A reset-free-only test would have passed.

The second implementation computed the right gradient but stacked activations across time,
so it grew 10.0× with T against BPTT's 9.9× — the correct answer with none of the benefit the
method exists for. Only measuring memory caught it.


---

## Framework comparison on SHD — 2026-08-09, NVIDIA T4 (Modal)

Every framework installed side by side and run in one container on one GPU, each in its own
subprocess so that no library is measured while another holds device memory. Published numbers
measured on other hardware are context, never evidence, so every figure here was re-measured.

### Protocol

`Linear(128, 128) -> LIF -> Linear(128, 128) -> LIF -> Linear(128, 20) -> LI`, no biases,
Adam at 5e-4, cross-entropy on the time-integrated readout with 0.3 label smoothing, fp32,
20 epochs at batch 256, T=256. Every framework receives bit-identical input arrays.

| framework | version | path exercised |
|---|---|---|
| jaxpike | this repo | `unroll`, `unroll_checkpointed`, `unroll_parallel` |
| SpikingJelly | 0.0.0.0.14 | multi-step, CuPy fused kernels |
| snnTorch | 1.0.0 | `snn.Leaky`, Python time loop |
| Norse | 1.1.0 | `LIFCell`, Python time loop |

Three deliberate concessions to the competition, so that a favourable result cannot be
dismissed as a rigged harness:

1. The leaky readout is evaluated in closed form for the PyTorch models. `sum_t v[t]` is
   linear in its input, so it collapses to a weighted sum over time — exact, and it spares
   them a 256-iteration Python loop that JAX fuses away.
2. SpikingJelly runs its fastest documented configuration, and the CuPy backend is verified at
   runtime rather than assumed: `SpikingJellyNet.check_backend` raises if it has silently
   fallen back to the Torch backend, which it otherwise does.
3. Each framework keeps its native decay parameterization rather than being forced to match.

### Training time and memory

| framework | 20 epochs | peak memory |
|---|---:|---:|
| SpikingJelly, multi-step + CuPy | **6.02 ± 0.00 s** | 792.1 MB |
| **jaxpike, `unroll`** | **8.12 ± 0.02 s** | 324.5 MB |
| **jaxpike, `unroll_checkpointed`** | 11.07 ± 0.02 s | **64.2 MB** |
| jaxpike, `unroll_parallel` | 15.80 ± 0.02 s | 288.1 MB |
| Norse | 252.21 ± 8.33 s | 737.3 MB |
| SpikingJelly, Torch backend | 260.62 ± 1.32 s | 696.3 MB |
| snnTorch | 347.18 ± 17.52 s | 675.8 MB |

jaxpike figures are steady state; compilation costs a further 14–19 s, paid once per shape,
and the PyTorch frameworks pay nothing equivalent. Memory is XLA's planned peak scratch for
jaxpike and `torch.cuda.max_memory_allocated` for the others; the two instruments are not
identical and the comparison should be read as an order of magnitude, not a ratio.

Anything that steps through time in Python loses by 31–43×, which is most of the field.
SpikingJelly's fused CuPy kernel is **1.35× faster than the best jaxpike path** and is not
beaten here.

### Accuracy

100 epochs, batch 256, T=256, hidden 128, single seed.

| model | test accuracy |
|---|---:|
| Spyx, as reported in arXiv 2402.18994 | 0.70 – 0.75 |
| jaxpike, matched model | **0.751** |
| jaxpike, reset-free neuron (`unroll_parallel`) | 0.609 |

Reset costs roughly 13 accuracy points on this task, which is the price of the parallel-in-time
path and the reason it is not the default.

### Where the remaining gap is

Ablation of a single training step, obtained by replacing one class of layer with a
passthrough:

| probe | time | share of step |
|---|---:|---:|
| full step | 12.19 ms | 100% |
| forward only | 3.55 ms | 29% |
| GEMMs only, forward and backward | 3.74 ms | 31% |
| neuron loops only, forward and backward | 10.12 ms | 83% |

The hoisted `Dense` layers compile to a single `f32[65536,128]` dot, confirmed in the emitted
HLO, so the matrix multiplications are not the problem. **83% of a step is the neuron time
loop and 71% is the backward pass.** A single LIF layer's forward takes 0.832 ms against a
bandwidth floor of about 0.34 ms for the 67 MB it must move, so roughly 2× of headroom sits
there — close to the size of the gap.

Capturing it requires fusing the time loop into one kernel, which is what
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) exists for. That is untested:
Pallas requires compute capability 8.0 or higher, its Triton backend refuses sm_75 outright,
and its Mosaic GPU backend targets Hopper. The T4 used throughout these benchmarks is sm_75.

### What is exhausted, and what is not

An automated search over execution strategies — every scan-unroll factor, every checkpoint
chunk size, and per-step rematerialization, each gated on agreeing with the reference in both
outputs and gradients before being timed — finds no configuration that beats the default. With
a 0.9% noise floor, everything except `unroll=1` and the parallel path is a statistical tie.
Further speed requires new code, not new settings.

Three optimizations that were expected to help and did not, recorded because a benchmark suite
that reports only its wins is marketing:

| change | expectation | measured |
|---|---|---|
| cast inputs per batch rather than per epoch | remove a 1.07 GB buffer | no change |
| skip the surrogate relaxation in the forward pass | remove 16.7M transcendentals | no change |
| recompute the neuron step instead of storing residuals | less memory traffic | 0.91×, but 22% less scratch |
