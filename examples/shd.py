"""Train a spiking network on Spiking Heidelberg Digits or Spiking Speech Commands.

Both are cochlea-model encodings of spoken audio across 700 input channels, and both have real
temporal structure rather than the rate coding of spiking MNIST. SHD is 20 classes of spoken
digits; SSC is 35 keyword classes and roughly ten times the data, which makes it the harder and
more informative of the two.

Reference accuracies from Cramer et al. (2020), who introduced both datasets: on SHD, ~48% for
a feedforward spiking net and ~71% for a recurrent one; on SSC, ~50% feedforward and ~57%
recurrent. Later work reaches higher with adaptive neurons and heavier training.

    python examples/shd.py --dataset shd --epochs 20
    python examples/shd.py --dataset ssc --epochs 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jaxpike as jp

N_CHANNELS = 700

# Both datasets share the cochlea encoding and channel count, and differ in class count and
# scale, so one script covers them by looking the class count up rather than hard-coding it.
DATASETS = {"shd": (jp.data.shd, 20), "ssc": (jp.data.ssc, 35)}


def load(split: str, root: Path, *, timesteps: int, dataset: str = "shd"):
    """Load one split of `dataset` as dense `(N, T, channels)` spikes and integer labels."""
    loader, _ = DATASETS[dataset]
    data = loader(split, root, timesteps=timesteps)
    return data.inputs, data.labels


def build(
    hidden: int,
    key,
    *,
    tau: float,
    threshold: float,
    recurrent: bool = False,
    n_classes: int = 20,
):
    """Feedforward by default; `recurrent` adds a self-connection on the hidden layer.

    The recurrent variant is the architecture Cramer et al. report ~71% with, against ~48%
    feedforward: the hidden layer's own spikes are fed back through a learned weight, giving
    the network a memory of its recent activity that a feedforward net simply does not have.

    The recurrent weight is initialised much smaller than the feedforward ones. Its output is
    summed into the same membrane on the next step, so a gain sized for feedforward drive
    makes the loop self-amplifying and the network saturates within a few timesteps.
    """
    gain = jp.lif_gain(tau)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    if not recurrent:
        return jp.Sequential(
            jp.Dense(N_CHANNELS, hidden, key=k1, gain=gain),
            jp.LinearLIF(tau=tau, threshold=threshold),
            jp.Dense(hidden, hidden, key=k2, gain=gain),
            jp.LinearLIF(tau=tau, threshold=threshold),
            jp.Dense(hidden, n_classes, key=k3, gain=gain),
            jp.LeakyIntegrator(tau=tau),
        )
    return jp.Graph(
        nodes={
            "w_in": jp.Dense(N_CHANNELS, hidden, key=k1, gain=gain),
            "h1": jp.LIF(tau=tau, threshold=threshold, reset="subtract"),
            "w_rec": jp.Dense(hidden, hidden, key=k2, gain=0.2),
            "w_h2": jp.Dense(hidden, hidden, key=k3, gain=gain),
            "h2": jp.LIF(tau=tau, threshold=threshold, reset="subtract"),
            "w_out": jp.Dense(hidden, n_classes, key=k4, gain=gain),
            "out": jp.LeakyIntegrator(tau=tau),
        },
        edges=[
            ("input", "w_in"),
            ("w_in", "h1"),
            ("h1", "w_rec"),
            ("w_rec", "h1"),
            ("h1", "w_h2"),
            ("w_h2", "h2"),
            ("h2", "w_out"),
            ("w_out", "out"),
        ],
        output="out",
    )


def augment(xs, key, *, max_shift: int, drop: float):
    """Random time shift plus input spike dropout.

    SHD has only 8156 training samples and the models here reach 99.9% training accuracy, so
    the ceiling is memorization rather than capacity. Both transforms are label-preserving:
    a spoken digit shifted a few milliseconds is the same digit, and dropping a fraction of
    input spikes mimics the variability of the cochlear model that generated them.
    """
    shift_key, drop_key = jax.random.split(key)
    if max_shift:
        offset = jax.random.randint(shift_key, (), -max_shift, max_shift + 1)
        xs = jnp.roll(xs, offset, axis=0)
    if drop:
        keep = jax.random.uniform(drop_key, xs.shape) >= drop
        xs = xs * keep
    return xs


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
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="shd")
    ap.add_argument("--data", type=Path, default=None, help="defaults to data/<dataset>")
    ap.add_argument("--sequential", action="store_true", help="disable parallel-in-time")
    ap.add_argument("--recurrent", action="store_true", help="recurrent hidden layer (Graph)")
    ap.add_argument("--augment", action="store_true", help="time shift + input spike dropout")
    ap.add_argument("--shift", type=int, default=20, help="max random time shift, in steps")
    ap.add_argument("--drop", type=float, default=0.1, help="input spike dropout probability")
    args = ap.parse_args()

    root = args.data if args.data is not None else Path(f"data/{args.dataset}")
    n_classes = DATASETS[args.dataset][1]
    print(f"device: {jax.devices()[0]}   dataset: {args.dataset} ({n_classes} classes)")
    x_train, y_train = load("train", root, timesteps=args.timesteps, dataset=args.dataset)
    x_test, y_test = load("test", root, timesteps=args.timesteps, dataset=args.dataset)

    # Hold out a validation split for model selection. Picking the best epoch by test accuracy
    # fits the test set through the choice of epoch and reports a number that will not
    # reproduce. Select on validation, report test once, at the selected epoch.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y_train))
    n_val = max(args.batch, int(0.1 * len(order)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    print(
        f"train {x_train.shape} {x_train.dtype}  val {x_val.shape}  test {x_test.shape}  "
        f"density {x_train.mean():.4f}  host size {x_train.nbytes / 2**30:.1f} GiB"
    )

    # A recurrent network has a genuine cycle in time, so parallel-in-time does not apply.
    use_sequential = args.sequential or args.recurrent
    runner = jp.unroll if use_sequential else jp.unroll_parallel
    model = build(
        args.hidden,
        jax.random.key(0),
        tau=args.tau,
        threshold=args.threshold,
        n_classes=n_classes,
        recurrent=args.recurrent,
    )
    print(
        f"model: {'recurrent Graph' if args.recurrent else 'feedforward Sequential'}, "
        f"runner: {'sequential' if use_sequential else 'parallel-in-time'}"
    )
    optimizer = optax.adamw(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_fn(m, xs, labels):
        membrane, _ = runner(m, xs)
        logits = jp.max_membrane_logits(membrane)
        loss = jp.cross_entropy(logits, labels)
        # Hidden-layer rate regularization, off by default: enable if the net saturates.
        if args.rate_weight and not args.recurrent:
            hidden_spikes, _ = runner(jp.Sequential(*m.layers[:2]), xs)
            loss = loss + args.rate_weight * jp.rate_penalty(hidden_spikes, args.rate_target)
        return loss, jp.accuracy(logits, labels)

    step = jp.make_step(loss_fn, optimizer)

    @eqx.filter_jit
    def evaluate(m, xs, labels):
        membrane, _ = runner(m, xs)
        return jp.accuracy(jp.max_membrane_logits(membrane), labels)

    def evaluate_all(m, inputs, labels, key):
        scores = [
            float(evaluate(m, xs, ys))
            for xs, ys in jp.iterate_batches(inputs, labels, args.batch, key=key, shuffle=False)
        ]
        return sum(scores) / len(scores)

    header = (
        f"{'epoch':>6} {'train loss':>11} {'train acc':>10} "
        f"{'val acc':>8} {'test acc':>9} {'sec':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    best_val, test_at_best, best_epoch = 0.0, 0.0, -1
    for epoch in range(args.epochs):
        start = time.perf_counter()
        key = jax.random.key(epoch + 1)
        losses, accs = [], []
        for index, (xs, ys) in enumerate(jp.iterate_batches(x_train, y_train, args.batch, key=key)):
            if args.augment:
                xs = augment(
                    xs,
                    jax.random.fold_in(key, index),
                    max_shift=args.shift,
                    drop=args.drop,
                )
            model, opt_state, loss, acc = step(model, opt_state, xs, ys)
            losses.append(float(loss))
            accs.append(float(acc))

        val_acc = evaluate_all(model, x_val, y_val, key)
        test_acc = evaluate_all(model, x_test, y_test, key)
        if val_acc > best_val:
            best_val, test_at_best, best_epoch = val_acc, test_acc, epoch
        print(
            f"{epoch:>6} {sum(losses) / len(losses):>11.4f} "
            f"{sum(accs) / len(accs):>10.4f} {val_acc:>8.4f} {test_acc:>9.4f} "
            f"{time.perf_counter() - start:>7.1f}"
        )

    print(f"\nselected epoch {best_epoch} by validation accuracy ({best_val:.4f})")
    print(f"TEST ACCURACY: {test_at_best:.4f}")
    print("Cramer et al. 2020 reference: ~0.48 feedforward, ~0.71 recurrent.")


if __name__ == "__main__":
    main()
