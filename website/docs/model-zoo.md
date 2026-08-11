---
id: model-zoo
title: Model zoo
sidebar_position: 98
---

# Model zoo

Every result here was produced by a script in the repository, on an NVIDIA T4, and is reported
against the accuracy published by whoever introduced the dataset. Where a number falls short of
its reference it says so.

Model selection is on a held-out validation split, never on test. Choosing the best epoch by
test accuracy fits the test set through the choice of epoch and produces a number that does not
reproduce.

## Spiking Heidelberg Digits

20 classes of spoken digits, 700 cochlea channels, 8,156 training samples. Introduced by Cramer
et al. (2020), who report **~48% feedforward** and **~71% recurrent**.

| model | test accuracy | reference | command |
|---|---:|---:|---|
| Feedforward, `LinearLIF`, parallel-in-time | 0.626 | ~0.48 | `python examples/shd.py --epochs 60 --augment` |
| Recurrent `Graph`, `LIF` | **0.696** | ~0.71 | `python examples/shd.py --epochs 100 --recurrent --augment` |

The recurrent variant feeds the hidden layer's own spikes back through a learned weight, which
is the architecture Cramer et al. report ~71% with. A recurrent graph is a genuine cycle in
time, so it runs sequentially rather than parallel-in-time.

### Under the Spyx protocol

A second SHD configuration, matching the protocol published for Spyx (arXiv 2402.18994) so the
two are comparable: 128 input channels, 256 timesteps, hidden 128, Adam 5e-4, integral
cross-entropy with 0.3 label smoothing, 100 epochs. **Five seeds**, since seed-to-seed spread on
this task is about ±0.02 and a single run cannot be distinguished from its own variance.

| model | mean | sd | range | reference |
|---|---:|---:|---|---:|
| `unroll`, reset LIF | **0.7532** | 0.0292 | 0.7075 – 0.7822 | 0.70 – 0.75 |
| `unroll_parallel`, reset-free | 0.6135 | 0.0189 | 0.5874 – 0.6392 | — |

The mean sits at the upper edge of the published band with a range overlapping it. Reset costs
roughly 14 accuracy points, which is the price of the parallel-in-time path.

Reproduce with `modal run benchmarks/gpu/run_torch_comparison.py --suite accuracy --epochs 100`.

## Spiking Speech Commands

35 keyword classes, 700 cochlea channels, 67,920 training samples — the same encoding as SHD at
roughly ten times the scale. Cramer et al. report **~50% feedforward** and **~57% recurrent**.

| model | test accuracy | reference | command |
|---|---:|---:|---|
| Recurrent `Graph`, `LIF`, 30 epochs | **0.5449** | ~0.57 | `python examples/shd.py --dataset ssc --epochs 30 --recurrent --augment` |

Epoch 29 selected on validation at 0.5528. This lands between the published feedforward and
recurrent references and short of the recurrent one; 30 epochs on a dataset this size is a
modest budget, and the gap is most likely training length rather than the model.

The inputs are 11.1 GiB on the host at `timesteps=250`, which is why `jaxpike.data` keeps them
as NumPy `uint8` and moves one batch at a time. Staging the whole dataset on the accelerator, as
the SHD benchmark does, is not an option at this scale.

## What is not here

**No measured event-camera accuracy.** `examples/nmnist.py` trains a spiking convnet on
N-MNIST and its architecture is verified — correct output shape and a forward pass through
`unroll_parallel` on `(T, batch, 34, 34, 2)` — but no accuracy number has been produced,
because the `tonic` dataset dependency would not install in the benchmark image. The example
ships; the row does not, and will not until there is a number behind it.

**No adaptive-neuron or deep-architecture results.** `ALIF` and `Izhikevich` are implemented
and tested but have not been trained on a benchmark, so the published accuracies that use
adaptive thresholds are not reproduced here.
