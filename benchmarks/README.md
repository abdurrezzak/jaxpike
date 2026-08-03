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
