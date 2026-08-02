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
the total work in exchange for `O(log T)` depth instead of `O(T)`, which is a losing trade on
a CPU with a handful of cores and a winning one on hardware with thousands of lanes. This
experiment establishes correctness only. **The speedup is unproven until Phase 0b measures it
on a GPU, and should not be claimed until then.**
