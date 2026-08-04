"""Event-based dataset loaders.

Spiking datasets ship as event lists — `(timestamp, unit)` pairs — rather than tensors, and
every framework re-implements the same binning code to turn them into something a network can
consume. This module does it once. Needs `h5py` (``pip install "jaxpike[data]"``).

Two decisions are baked in:

**Arrays stay on the host.** A long-sequence spiking dataset is enormous when densified: SHD
at 1000 timesteps is 8156 by 1000 by 700, which is 22.8 GB in float32 and will not fit on most
accelerators alongside a model. `jaxpike.iterate_batches` moves one batch at a time.

**Spikes are stored as uint8, not float32.** They are binary, so float32 wastes four times the
memory and four times the host-to-device bandwidth every batch.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ZENKE_URL = "https://zenkelab.org/datasets/{name}.h5.gz"


@dataclass(frozen=True)
class Dataset:
    """A densified event dataset.

    `inputs` is `(samples, timesteps, units)` uint8 on the host; `labels` is `(samples,)` int32.
    """

    inputs: np.ndarray
    labels: np.ndarray
    name: str
    split: str

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_classes(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def density(self) -> float:
        return float(self.inputs.mean())

    def __repr__(self) -> str:
        gib = self.inputs.nbytes / 2**30
        return (
            f"Dataset({self.name}/{self.split}: {len(self)} samples, "
            f"{self.inputs.shape[1]} timesteps, {self.inputs.shape[2]} units, "
            f"{self.n_classes} classes, density {self.density:.4f}, {gib:.2f} GiB host)"
        )


def _download(url: str, target: Path) -> Path:
    """Fetch and gunzip `url` to `target`, unless it is already there."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.with_suffix(target.suffix + ".gz")
    print(f"downloading {target.name} ...", flush=True)
    urllib.request.urlretrieve(url, archive)
    with gzip.open(archive, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    archive.unlink()
    return target


def bin_events(times, units, labels, *, timesteps: int, n_units: int, duration: float):
    """Densify event lists onto a fixed `timesteps` grid.

    Multiple events landing in the same bin are collapsed to a single spike rather than
    accumulated: a count above one is an artifact of the chosen resolution, and the neurons
    take binary input.
    """
    out = np.zeros((len(labels), timesteps, n_units), dtype=np.uint8)
    for index, (event_times, event_units) in enumerate(zip(times, units, strict=True)):
        bins = np.clip(
            (np.asarray(event_times) / duration * timesteps).astype(np.int64), 0, timesteps - 1
        )
        out[index][bins, np.asarray(event_units, dtype=np.int64)] = 1
    return out


def _load_zenke(
    name: str, split: str, root: Path, *, timesteps: int, n_units: int, duration: float
) -> Dataset:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('event datasets need h5py: pip install "jaxpike[data]"') from exc

    path = _download(ZENKE_URL.format(name=f"{name}_{split}"), root / f"{name}_{split}.h5")
    with h5py.File(path, "r") as handle:
        times = handle["spikes"]["times"][:]
        units = handle["spikes"]["units"][:]
        labels = np.asarray(handle["labels"][:], dtype=np.int32)
    inputs = bin_events(
        times, units, labels, timesteps=timesteps, n_units=n_units, duration=duration
    )
    return Dataset(inputs=inputs, labels=labels, name=name, split=split)


def shd(split: str = "train", root: Path | str = "data/shd", *, timesteps: int = 250) -> Dataset:
    """Spiking Heidelberg Digits: 20 classes, 700 cochlear channels, ~1 s utterances.

    The field's standard temporal benchmark. Published baselines from Cramer et al. (2020):
    ~0.48 test accuracy feedforward, ~0.71 recurrent.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return _load_zenke("shd", split, Path(root), timesteps=timesteps, n_units=700, duration=1.0)


def ssc(split: str = "train", root: Path | str = "data/ssc", *, timesteps: int = 250) -> Dataset:
    """Spiking Speech Commands: 35 classes, 700 channels. Same encoding as SHD, much larger."""
    if split not in ("train", "test", "valid"):
        raise ValueError(f"split must be 'train', 'test' or 'valid', got {split!r}")
    return _load_zenke("ssc", split, Path(root), timesteps=timesteps, n_units=700, duration=1.0)


__all__ = ["Dataset", "bin_events", "shd", "ssc"]
