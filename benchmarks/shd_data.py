"""SHD loading shared by every framework in the benchmark suite.

Kept free of framework imports so the PyTorch benchmarks can use it without pulling JAX onto
the GPU alongside torch.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 20
SHD_CHANNELS = 700


def downsample_channels(dense: np.ndarray, out_channels: int) -> np.ndarray:
    """Fold 700 cochlear channels into `out_channels`, matching tonic's spatial Downsample.

    tonic maps each event's channel to ``floor(x * out/700)`` and the rasterizer then clips
    the count to 1, so a channel group fires if any of its members did -- a max over groups.
    """
    index = (np.arange(SHD_CHANNELS) * out_channels // SHD_CHANNELS).astype(np.int64)
    out = np.zeros((*dense.shape[:2], out_channels), dtype=np.uint8)
    np.maximum.at(out.transpose(2, 0, 1), index, dense.transpose(2, 0, 1))
    return out


def load(split: str, root: str, *, timesteps: int, channels: int):
    from jaxpike.data import shd

    dataset = shd(split, root, timesteps=timesteps)
    return downsample_channels(dataset.inputs, channels), dataset.labels


def fake(n: int, timesteps: int, channels: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    inputs = (rng.random((n, timesteps, channels)) < 0.05).astype(np.uint8)
    labels = rng.integers(0, N_CLASSES, size=n).astype(np.int32)
    return inputs, labels
