---
id: data
title: Datasets
sidebar_position: 6
---

# Datasets

`jaxpike.data` needs h5py: `pip install "jaxpike[data]"`.

Spiking datasets ship as event lists — `(timestamp, unit)` pairs — rather than tensors, and
every framework re-implements the same binning code to turn them into something a network can
consume. This module does it once.

## Loaders

```python
train = jp.data.shd("train", "data/shd", timesteps=250)   # 20 classes, 700 channels
test  = jp.data.shd("test", "data/shd", timesteps=250)

train = jp.data.ssc("train", "data/ssc", timesteps=250)   # 35 classes, same encoding
```

Both download from the Zenke lab on first use and cache under `root`. SHD splits are `"train"`
and `"test"`; SSC adds `"valid"`.

**SHD** is the field's standard temporal benchmark: 20 spoken digits (English and German), ~1 s
utterances over 700 cochlear channels. Published baselines from Cramer et al. (2020) are ~0.48
test feedforward and ~0.71 recurrent. **SSC** is Spiking Speech Commands — same encoding, 35
classes, much larger.

## `Dataset`

```python
Dataset(inputs, labels, name, split)
```

`inputs` is `(samples, timesteps, units)` **uint8 on the host**; `labels` is `(samples,)` int32.
Also exposes `len(dataset)`, `n_classes`, and `density`, and its `repr` prints the shape,
density and host size, which is the quickest sanity check that binning did what you expected:

```
Dataset(shd/train: 8156 samples, 250 timesteps, 700 units, 20 classes, density 0.0xxx, 1.33 GiB host)
```

## Two decisions baked in

**Arrays stay on the host.** A long-sequence spiking dataset is enormous when densified: SHD at
1000 timesteps is 8156 × 1000 × 700, which is 22.8 GB in float32 and will not fit on most
accelerators alongside a model. `jp.iterate_batches` moves one batch at a time.

**Spikes are stored as uint8, not float32.** They are binary, so float32 costs four times the
memory and four times the host-to-device bandwidth every batch. On a long-sequence run that is
the difference between being compute-bound and transfer-bound, and it is large enough to make a
2.4× speedup measure as 1.56×.

## Binning your own events

```python
jp.data.bin_events(times, units, labels, *, timesteps, n_units, duration) -> np.ndarray
```

Densifies event lists onto a fixed `timesteps` grid, returning `(samples, timesteps, n_units)`
uint8. `duration` is the recording length in the same units as `times`.

Multiple events landing in the same bin are collapsed to a single spike rather than accumulated.
A count above one is an artifact of the chosen time resolution, not something the recording
distinguishes, and the neurons take binary input.

## Not yet included

N-MNIST and DVS Gesture are the obvious next two. Both need a different loader shape — spatial
event streams `(samples, timesteps, height, width, channels)` rather than 1-D channels. Bin them
yourself to that shape as uint8 on the host and `iterate_batches` will handle the rest.
