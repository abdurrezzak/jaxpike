"""Train a spiking convnet on N-MNIST.

N-MNIST is MNIST recorded with an event camera: each digit is captured by a sensor that
reports brightness *changes* at 34x34 resolution across two polarity channels, so the input is
already a spike train rather than an image that has to be rate-coded into one. It is the
standard first benchmark for event-driven vision.

Events are binned into `--timesteps` frames, giving `(time, batch, 34, 34, 2)` in the NHWC
layout the convolutions expect. Reference accuracies are in the high 90s for convolutional
spiking networks; a small network trained briefly lands in the 90s.

Requires `tonic` for the dataset:

    pip install tonic
    python examples/nmnist.py --epochs 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import optax

import jaxpike as jp

SENSOR = (34, 34, 2)
N_CLASSES = 10


def load(split: str, root: Path, *, timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    """Events binned into dense frames, `(N, T, 34, 34, 2)` uint8 and integer labels.

    Kept on the host as uint8: at `timesteps=10` the training split is 1.4 GiB dense, and
    float32 on the accelerator would be four times that before training starts.
    """
    import tonic

    dataset = tonic.datasets.NMNIST(save_to=str(root), train=split == "train")
    to_frame = tonic.transforms.ToFrame(
        sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=timesteps
    )

    inputs = np.zeros((len(dataset), timesteps, *SENSOR), dtype=np.uint8)
    labels = np.zeros(len(dataset), dtype=np.int32)
    for index, (events, label) in enumerate(dataset):
        # tonic emits (time, polarity, height, width); the convolutions here are NHWC.
        frames = to_frame(events).transpose(0, 2, 3, 1)
        inputs[index] = np.minimum(frames, 1).astype(np.uint8)
        labels[index] = label
    return inputs, labels


def build(key, *, tau: float, threshold: float, channels: int = 32):
    """Two convolutional stages then a dense readout, all reset-free so time parallelizes."""
    k1, k2, k3 = jax.random.split(key, 3)
    gain = jp.lif_gain(tau)
    # 34 -> 17 -> 8 after two 2x pools, with 'same' convolutions preserving resolution.
    flat = 8 * 8 * channels * 2
    return jp.Sequential(
        jp.Conv2d(2, channels, 3, key=k1, gain=gain),
        jp.LinearLIF(tau=tau, threshold=threshold),
        jp.Pool2d(2),
        jp.Conv2d(channels, channels * 2, 3, key=k2, gain=gain),
        jp.LinearLIF(tau=tau, threshold=threshold),
        jp.Pool2d(2),
        jp.Flatten(),
        jp.Dense(flat, N_CLASSES, key=k3, gain=gain),
        jp.LeakyIntegrator(tau=tau),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", type=Path, default=Path("data/nmnist"))
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
    model = build(jax.random.key(0), tau=args.tau, threshold=args.threshold, channels=args.channels)
    optimizer = optax.adam(args.lr)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    def loss_fn(p, xs, labels):
        membrane, _ = runner(eqx.combine(p, static), xs)
        return jp.cross_entropy(jp.max_membrane_logits(membrane), labels)

    @eqx.filter_jit
    def train_step(p, state, xs, labels):
        loss, grads = jax.value_and_grad(loss_fn)(p, xs, labels)
        updates, state = optimizer.update(grads, state, p)
        return eqx.apply_updates(p, updates), state, loss

    @eqx.filter_jit
    def evaluate(p, xs, labels):
        membrane, _ = runner(eqx.combine(p, static), xs)
        return jp.accuracy(jp.max_membrane_logits(membrane), labels)

    print(f"{'epoch':>6} {'loss':>10} {'test acc':>10} {'sec':>8}")
    for epoch in range(args.epochs):
        start = time.perf_counter()
        losses = []
        batches = jp.iterate_batches(x_train, y_train, args.batch, key=jax.random.key(epoch))
        for xs, labels in batches:
            params, opt_state, loss = train_step(params, opt_state, xs, labels)
            losses.append(float(loss))

        correct = total = 0
        eval_key = jax.random.key(0)
        for xs, labels in jp.iterate_batches(
            x_test, y_test, args.batch, key=eval_key, shuffle=False
        ):
            correct += float(evaluate(params, xs, labels)) * len(labels)
            total += len(labels)
        print(
            f"{epoch:>6} {np.mean(losses):>10.4f} {correct / total:>10.4f} "
            f"{time.perf_counter() - start:>8.1f}"
        )

    print(f"TEST ACCURACY: {correct / total:.4f}")


if __name__ == "__main__":
    main()
