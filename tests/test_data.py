"""Dataset loader tests.

The download-backed loaders are skipped unless the data is already cached, so CI stays fast
and offline. `bin_events` is the part with real logic and is tested directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from jaxpike.data import Dataset, bin_events, shd, ssc

CACHED = Path("data/shd/shd_test.h5").exists()


def test_bin_events_places_spikes_in_the_right_bins():
    times = [np.array([0.0, 0.5, 0.99])]
    units = [np.array([0, 3, 7])]
    out = bin_events(times, units, np.array([0]), timesteps=10, n_units=8, duration=1.0)
    assert out.shape == (1, 10, 8)
    assert out[0, 0, 0] == 1
    assert out[0, 5, 3] == 1
    assert out[0, 9, 7] == 1
    assert out.sum() == 3


def test_bin_events_collapses_duplicates_rather_than_counting():
    """Two events in one bin is a resolution artifact, not a magnitude the recording captured."""
    times = [np.array([0.10, 0.11, 0.12])]
    units = [np.array([2, 2, 2])]
    out = bin_events(times, units, np.array([0]), timesteps=10, n_units=4, duration=1.0)
    assert out[0, 1, 2] == 1
    assert out.max() == 1


def test_bin_events_clamps_out_of_range_times():
    times = [np.array([-0.5, 1.5])]
    units = [np.array([0, 1])]
    out = bin_events(times, units, np.array([0]), timesteps=10, n_units=2, duration=1.0)
    assert out[0, 0, 0] == 1 and out[0, 9, 1] == 1


def test_bin_events_returns_uint8():
    out = bin_events(
        [np.array([0.5])], [np.array([1])], np.array([0]), timesteps=4, n_units=3, duration=1.0
    )
    assert out.dtype == np.uint8


@pytest.mark.parametrize(("loader", "split"), [(shd, "validation"), (ssc, "nope")])
def test_bad_split_is_rejected(loader, split):
    with pytest.raises(ValueError, match="split must be"):
        loader(split)


def test_dataset_summary_fields():
    inputs = np.zeros((5, 10, 4), dtype=np.uint8)
    inputs[0, 0, 0] = 1
    ds = Dataset(inputs=inputs, labels=np.array([0, 1, 2, 1, 0], np.int32), name="x", split="y")
    assert len(ds) == 5
    assert ds.n_classes == 3
    assert ds.density == pytest.approx(1 / 200)
    assert "5 samples" in repr(ds) and "3 classes" in repr(ds)


@pytest.mark.skipif(not CACHED, reason="SHD not cached; skipping the download-backed check")
def test_cached_shd_loads_with_the_expected_shape():
    ds = shd("test", timesteps=100)
    assert ds.inputs.shape == (len(ds), 100, 700)
    assert ds.inputs.dtype == np.uint8
    assert ds.n_classes == 20
    assert 0.0 < ds.density < 0.5
