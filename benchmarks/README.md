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
each other, and the wrong one for predicting whether a model fits on a given GPU. The GPU section
below cross-checks it against real device measurements.

---

## Memory and parallel-in-time — 2026-08-01, Apple Silicon CPU, JAX 0.11.0

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

**66.6× less memory at T=5000.**

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

A practical consequence: pure-JAX rematerialization already captures most of the available
memory win, so fused kernels must justify themselves on **bandwidth and speed** rather than on
memory, and must be benchmarked against the checkpointed path rather than the naive one.

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

Worst deviation 5.96e-07 at T=8192, pure float32
accumulation-order noise, and — the measure that actually matters — the binary spike trains
are *identical* at every length, so no threshold crossing was flipped.

**The associative scan is 3–5× slower here, and that is expected.** It performs roughly 2×
the total work in exchange for `O(log T)` depth instead of `O(T)`: a losing trade on a CPU
with a handful of cores, a winning one on hardware with thousands of lanes. Correctness is
what this run establishes. For the speedup, see the GPU results below.

---

## The same measurements on GPU — 2026-08-03, NVIDIA T4 (Modal), JAX 0.11.0

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

This is an algorithmic win that needs no custom kernels, no custom VJP and no hand-tuning.

**The limit, stated plainly.** This is the *reset-free* case, where the recurrence is linear.
Reset makes it nonlinear and this exact method no longer applies — that is what chunked scan
and the DEER-style tier of the plan are for, and neither is proven yet. Reset-free PSN-style
neurons are a real published model class, so the result is directly usable, but it is not yet
a general LIF result and must not be reported as one.


---

## End-to-end network results — 2026-08-03, NVIDIA T4 (Modal)

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

### Accuracy across seeds

100 epochs, batch 256, T=256, hidden 128, five seeds each. Accuracy on this task varies by
roughly ±0.02 between seeds, which is large enough that a single run cannot be distinguished
from its own variance.

| model | seeds | mean | sd | range |
|---|---:|---:|---:|---|
| jaxpike, `unroll` | 5 | **0.7532** | 0.0292 | 0.7075 – 0.7822 |
| SpikingJelly, multi-step + CuPy | 5 | 0.6754 | 0.0205 | 0.6440 – 0.6938 |
| jaxpike, `unroll_parallel` (reset-free) | 5 | 0.6135 | 0.0189 | 0.5874 – 0.6392 |
| Spyx, as reported in arXiv 2402.18994 | — | 0.70 – 0.75 | — | — |

jaxpike's mean sits at the upper edge of the band published for Spyx, with a range that
overlaps it.

**The SpikingJelly row is not a like-for-like accuracy comparison and should not be read as
one.** Its neuron learns a single shared `tau`, which is its native parameterization, while
jaxpike and Spyx learn a per-neuron `beta`. That is fewer decay parameters and a lower ceiling.
The concession was made so the *speed* comparison would exercise SpikingJelly's fastest path;
it makes the accuracy column a comparison of two different models.

Reset costs roughly 14 accuracy points on this task, which is the price of the parallel-in-time
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
