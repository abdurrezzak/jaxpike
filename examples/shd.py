"""Train a spiking network on the Spiking Heidelberg Digits.

SHD is the field's standard temporal benchmark: 20 classes (digits 0-9 spoken in English and
German), converted to spike trains across 700 input channels by a cochlea model. Unlike
rate-coded MNIST the temporal structure is real, and the sequences are long -- which is where
`unroll_parallel` earns its keep.

Reference accuracies from Cramer et al. (2020), who introduced the dataset: ~48% for a
feedforward spiking net, ~71% for a recurrent one. Later work reaches the 80s-90s with
adaptive neurons and heavier training. A plain feedforward net in the high 40s to 60s is the
honest target here.

    python examples/shd.py --epochs 20
    modal run benchmarks/gpu/run_modal.py --bench shd     # on a GPU
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import time
import urllib.request
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import optax

import jaxpike as jp

URL = "https://zenkelab.org/datasets/{}.h5.gz"
N_CHANNELS, N_CLASSES = 700, 20


def download(split: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"shd_{split}.h5"
    if target.exists():
        return target
    gz = target.with_suffix(".h5.gz")
    print(f"downloading shd_{split} ...", flush=True)
    urllib.request.urlretrieve(URL.format(f"shd_{split}"), gz)
    with gzip.open(gz, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    gz.unlink()
    return target


def load(split: str, root: Path, *, timesteps: int, duration: float = 1.0):
    """Bin events into a dense `(samples, timesteps, channels)` array of spike counts.

    SHD ships as event lists (times, units). Binning to a fixed grid is what every published
    baseline does; `timesteps` sets the temporal resolution and directly sets sequence length.
    """
    import h5py

    path = download(split, root)
    with h5py.File(path, "r") as f:
        times = f["spikes"]["times"][:]
        units = f["spikes"]["units"][:]
        labels = np.asarray(f["labels"][:], dtype=np.int32)

    # uint8, not float32: these are binary spikes, so float32 wastes 4x the memory and 4x the
    # host-to-device bandwidth. At 1000 timesteps that is the difference between a 22.8 GB
    # dataset and a 5.7 GB one, and between being transfer-bound and compute-bound.
    out = np.zeros((len(labels), timesteps, N_CHANNELS), dtype=np.uint8)
    for i, (ts, us) in enumerate(zip(times, units, strict=True)):
        bins = np.clip((np.asarray(ts) / duration * timesteps).astype(np.int64), 0, timesteps - 1)
        # Assignment rather than accumulation: multiple events landing in one bin is a
        # resolution artifact, and the model takes binary spikes.
        out[i][bins, np.asarray(us, dtype=np.int64)] = 1
    # Deliberately left on the host. See jaxpike.training.iterate_batches: at 1000 timesteps
    # this array is 22.8 GB and will not fit on the GPU alongside the model.
    return out, labels


def build(hidden: int, key, *, tau: float, threshold: float):
    gain = jp.lif_gain(tau)
    k1, k2, k3 = jax.random.split(key, 3)
    return jp.Sequential(
        jp.Dense(N_CHANNELS, hidden, key=k1, gain=gain),
        jp.LinearLIF(tau=tau, threshold=threshold),
        jp.Dense(hidden, hidden, key=k2, gain=gain),
        jp.LinearLIF(tau=tau, threshold=threshold),
        jp.Dense(hidden, N_CLASSES, key=k3, gain=gain),
        jp.LeakyIntegrator(tau=tau),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=250)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rate-target", type=float, default=0.05)
    ap.add_argument("--rate-weight", type=float, default=0.0)
    ap.add_argument("--data", type=Path, default=Path("data/shd"))
    ap.add_argument("--sequential", action="store_true", help="disable parallel-in-time")
    args = ap.parse_args()

    print(f"device: {jax.devices()[0]}")
    x_train, y_train = load("train", args.data, timesteps=args.timesteps)
    x_test, y_test = load("test", args.data, timesteps=args.timesteps)
    print(
        f"train {x_train.shape} {x_train.dtype}  test {x_test.shape}  "
        f"density {x_train.mean():.4f}  host size {x_train.nbytes / 2**30:.1f} GiB"
    )

    runner = jp.unroll if args.sequential else jp.unroll_parallel
    model = build(args.hidden, jax.random.key(0), tau=args.tau, threshold=args.threshold)
    optimizer = optax.adamw(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(m, xs, labels):
        membrane, _ = runner(m, xs)
        logits = jp.max_membrane_logits(membrane)
        loss = jp.cross_entropy(logits, labels)
        # Hidden-layer rate regularization, off by default: enable if the net saturates.
        if args.rate_weight:
            hidden_spikes, _ = runner(jp.Sequential(*m.layers[:2]), xs)
            loss = loss + args.rate_weight * jp.rate_penalty(hidden_spikes, args.rate_target)
        return loss, jp.accuracy(logits, labels)

    step = jp.make_step(loss_fn, optimizer)

    @eqx.filter_jit
    def evaluate(m, xs, labels):
        membrane, _ = runner(m, xs)
        return jp.accuracy(jp.max_membrane_logits(membrane), labels)

    print(f"\n{'epoch':>6} {'train loss':>11} {'train acc':>10} {'test acc':>9} {'sec':>7}")
    print("-" * 48)
    best = 0.0
    for epoch in range(args.epochs):
        start = time.perf_counter()
        key = jax.random.key(epoch + 1)
        losses, accs = [], []
        for xs, ys in jp.iterate_batches(x_train, y_train, args.batch, key=key):
            model, opt_state, loss, acc = step(model, opt_state, xs, ys)
            losses.append(float(loss))
            accs.append(float(acc))

        test_accs = [
            float(evaluate(model, xs, ys))
            for xs, ys in jp.iterate_batches(x_test, y_test, args.batch, key=key, shuffle=False)
        ]
        test_acc = sum(test_accs) / len(test_accs)
        best = max(best, test_acc)
        print(
            f"{epoch:>6} {sum(losses) / len(losses):>11.4f} "
            f"{sum(accs) / len(accs):>10.4f} {test_acc:>9.4f} "
            f"{time.perf_counter() - start:>7.1f}"
        )

    print(f"\nbest test accuracy: {best:.4f}")
    print("Cramer et al. 2020 reference: ~0.48 feedforward, ~0.71 recurrent.")


if __name__ == "__main__":
    main()
