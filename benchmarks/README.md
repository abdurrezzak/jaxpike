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

## Head-to-head against Spyx on SHD — 2026-08-04, NVIDIA T4 (Modal)

Spyx is the closest competitor: also JAX, also JIT-compiled, also benchmarked on SHD. Table 1
of [arXiv 2402.18994](https://arxiv.org/abs/2402.18994) reports 100-epoch training times for
Spyx, snnTorch and mlGeNN, which makes it the one published comparison that can be contested
without a PyTorch-versus-JAX confound.

### Why these runs re-measure Spyx instead of quoting their table

Their numbers are an **RTX A6000 with 48 GB**. Ours are a **T4 with 16 GB**, the largest GPU
Modal's free tier allows. Quoting one beside the other would compare hardware. So
`benchmarks/gpu/run_spyx_comparison.py` installs both libraries in **one container** and runs
them on **one GPU**, and `benchmarks/spyx_reference.py` is Spyx's own benchmark code from
`research/paper/SHD_jax.ipynb` rather than a re-implementation, pinned to `spyx==0.1.19` —
the release contemporary with the paper. Later Spyx moved to Flax NNX and that API no longer
exists. Both libraries receive bit-identical input arrays, so neither pays a loader cost the
other avoids.

### The protocol, taken from their notebook rather than the paper prose

SHD downsampled to **128 input channels** at **256 timesteps**, binary-rasterized;
`Linear(128, H) -> LIF -> Linear(H, H) -> LIF -> Linear(H, 20) -> LI`, all Linear layers
biasless; Adam at 5e-4; integral cross-entropy with 0.3 label smoothing; 100 epochs; fp32;
the whole dataset staged on the accelerator, with epochs and batches both `lax.scan`.

Their LIF is **not** jaxpike's `LIF`, and substituting one would have measured the model
rather than the implementation. Two differences matter: the spike is read from the membrane
*before* this step's input, so it lags a timestep, and the input carries no `(1 - alpha)`
normalization, so weights are not in threshold units. `benchmarks/spyx_shd.py` therefore
implements `SpyxLIF`, `SpyxLinearLIF`, `SpyxLI` and their arctan surrogate directly against
the state contract.

### Accuracy against Spyx: matched model, matched result

Like for like — both models have reset, both run the same protocol:

| | test accuracy |
|---|---:|
| Spyx, as reported in the paper | 0.70 – 0.75 |
| **jaxpike, matched model (`SpyxLIF`)** | **0.751** |

jaxpike's own `examples/shd.py` scores 0.626 feedforward, but that is a *different
experiment*, not a worse library: 700 input channels, a single `tau=20`, threshold 0.5 and
AdamW at 2e-3. Spyx's benchmark downsamples to 128 channels and learns a per-neuron `beta`
initialized around 0.5 — a much leakier, heterogeneous membrane. Matching the specification
closed the gap outright.

### What reset costs, measured separately

This is a jaxpike-internal ablation with **no Spyx counterpart** — the benchmarked release
(0.1.19) ships no reset-free neuron. It is reported apart from the table above because reset
is what the parallel-in-time path cannot have, so it prices the fast path:

| model | test accuracy | layer firing rates |
|---|---:|---|
| `SpyxLIF` — with reset, sequential only | **0.751** | — |
| `SpyxPSU` — reset-free, integrate-then-spike, parallel | 0.609 | 0.090, 0.065 |
| `SpyxLinearLIF` — reset-free with the spike lag kept, parallel | 0.627 | 0.091, 0.067 |

**Removing reset costs about 13 accuracy points here, and two obvious explanations are both
wrong.**

*Not saturation.* The standard reset-free failure is a neuron that cannot depress itself,
fires every step, and lands where the surrogate gradient is flat. Measured rates are 6–9%,
comfortably inside the healthy band — the networks are not saturated, they are simply less
discriminative.

*Not a spike-timing artifact.* The first ablation removed reset but kept Spyx's one-step spike
lag, so it also discarded the current step's input; `SpyxPSU` integrates before spiking, which
is how Spyx's own `PSU_LIF` and jaxpike's `LinearLIF` are written. Correcting it did not
recover the points — it scored 1.8 points *lower*, which on a single seed is noise.

So the cost looks like a genuine property of the model rather than an implementation detail,
which agrees with Spyx's own description of their `PSU_LIF`: "removing the reset is a
deliberate accuracy/parallelism trade-off". **Any speed number taken from a reset-free row has
to carry this accuracy alongside it.**

Two caveats. These are **single-seed runs with no error bars**, so differences of a couple of
points mean nothing. And no hyperparameter search was run for the reset-free variants — they
inherit a threshold and `beta` initialization tuned by Spyx for a neuron that resets, and the
PSN literature does reach strong SHD accuracy with reset-free neurons, so the gap may narrow
under a search that has not been done.

### Parallel-in-time is no longer a unique differentiator

Spyx 1.0.0 ships `PSU_LIF` and `AssociativeLIF`: a reset-free neuron evaluated with
`jax.lax.associative_scan` in `O(log T)` depth, marked experimental. It is absent from 0.1.19,
the release the paper benchmarked, so the reproduction above is unaffected — but as of their
current release the reset-free parallel-scan idea is implemented on both sides, and any claim
that jaxpike alone has it would be false.

### Correctness of the ported model

Checked before any timing was recorded, since a fast wrong answer is worth nothing:
parallel-in-time matches sequential to **2.4e-07** on the reset-free model (float32
accumulation order, consistent with §4), checkpointed matches sequential exactly, and the
reset-based model is **refused** by `unroll_parallel` by name rather than silently
mis-computed.

### Training speed, both libraries in one container on one T4

Timings are 1 warm-up run plus 2 timed runs, reported as mean ± sd. Every ratio compares rows
measured inside the same container; numbers from different containers are never divided,
because Spyx's run-to-run spread on a T4 turned out to be large enough to invent a result
(±21.6 s at batch 256, against jaxpike's ±0.4 s).

**Batch size**, at T=256, hidden 128, 20 epochs:

| batch | Spyx 0.1.19 | jaxpike sequential | jaxpike checkpointed | jaxpike parallel |
|---:|---:|---:|---:|---:|
| 64 | 110.1 ± 2.6 | 73.9 ± 0.7 (1.5×) | 93.7 ± 0.6 (1.2×) | **33.8 ± 0.0 (3.3×)** |
| 128 | 89.4 ± 2.2 | 49.0 ± 0.9 (1.8×) | 59.5 ± 0.6 (1.5×) | 43.0 ± 0.4 (2.1×) |
| 256 | 90.7 ± 21.6 | **38.2 ± 0.4 (2.4×)** | 40.1 ± 1.0 (2.3×) | 46.9 ± 1.9 (1.9×) |

**Sequence length**, at batch 256, hidden 128, 10 epochs:

| T | Spyx 0.1.19 | jaxpike sequential | jaxpike parallel | seq vs Spyx | par vs seq |
|---:|---:|---:|---:|---:|---:|
| 256 | 60.4 ± 0.1 | 23.1 ± 0.0 | 26.5 ± 0.0 | 2.6× | 0.87× |
| 512 | 182.5 ± 9.5 | 45.7 ± 0.2 | 46.7 ± 1.0 | 4.0× | 0.98× |
| 1,024 | 776.9 ± 35.8 | 98.4 ± 1.6 | **81.9 ± 1.0** | **7.9×** | **1.20×** |

**jaxpike is faster everywhere measured, on the matched model, at full accuracy** — and the
path doing the winning is the ordinary sequential one, not anything exotic.

**The advantage compounds with sequence length: 2.6× → 4.0× → 7.9×.** Per doubling of `T`,
Spyx costs 3.0× then 4.3× more while jaxpike costs 2.0× then 2.15×. That gap is structural
rather than tuning: their benchmark uses `hk.static_unroll`, which materializes a graph
proportional to `T`, where `lax.scan` compiles one loop body whatever `T` is.

### Separating compile cost from throughput

Solving the 10- and 20-epoch runs at batch 256 as two points on a line splits fixed from
marginal cost:

| | compile (fixed) | per epoch |
|---|---:|---:|
| Spyx | ~30 s | 3.04 s |
| jaxpike | ~8 s | **1.51 s** |

So the steady-state advantage is **2.0×**, and the larger headline ratios come from also
compiling ~3.7× faster. Warm-up runs sit within ~2 s of steady state for both libraries, so
compilation is not otherwise distorting the totals. Quoting the headline ratio without this
split would overstate the throughput difference.

### Parallel-in-time: a narrow win, and it has to buy its way in

Against jaxpike's own sequential path the associative scan is **0.87× at T=256, 0.98× at
T=512 and 1.20× at T=1024** — it crosses over around T≈512 — and separately it is 2.2× faster
at batch 64. The mechanism is consistent: the scan performs roughly twice the work to buy
`O(log T)` depth, which pays only when the GPU has lanes to spare, either because the batch
is small or because the sequence is long.

That win has to cover a 13-point accuracy loss from dropping reset, so at batch 256 and
T=1024 a 1.20× speedup does not obviously pay for itself. The regime where it clearly does is
small batches: 3.3× against Spyx at batch 64, where the accuracy cost is the only thing
standing against it.

### Memory, peak scratch as XLA plans it

| variant | T=256 | T=1,024 |
|---|---:|---:|
| checkpointed | **28.9 MB** | — |
| sequential | 206.9 MB | 812.9 MB |
| parallel | 256.1 MB | 1,012.3 MB |

`unroll_checkpointed` is the quiet result of this comparison: **7.2× less scratch than the
sequential path at essentially the same speed** (40.1 s against 38.2 s at batch 256), and
Spyx has no equivalent — their paper notes they could not report memory at all. Sequential
scratch grows linearly in `T` as `O(T·B·N)` predicts, so at long sequences it is
checkpointing rather than parallel-in-time that keeps a model on a 16 GB card. No checkpointed
row was measured at T=1024, which in hindsight is the row most worth having.

### What this comparison does and does not establish

It establishes that jaxpike trains the same model faster than Spyx on the same GPU, by 1.5×
to 7.9× depending on configuration, with the gap widening in `T`, and that it does so on the
plain sequential path with no accuracy sacrifice.

It does not establish that parallel-in-time is the reason. It is not: it wins only at small
batch or long sequence, it costs 13 accuracy points, and Spyx 1.0.0 now ships its own
associative-scan neuron. Nor does it establish anything about snnTorch, SpikingJelly or
mlGeNN, none of which were re-run here.

Every accuracy figure is a **single seed with no error bars**, and Spyx's timings are unstable
on a T4 in a way jaxpike's are not — which is itself unexplained, and worth understanding
before leaning on any specific ratio.
