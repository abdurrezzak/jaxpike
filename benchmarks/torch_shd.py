"""The same SHD training run implemented in snnTorch, SpikingJelly and Norse.

The network, loss, optimizer, dtype and timing protocol match `benchmarks/spyx_shd.py`
op-for-op, so the only thing that differs between runs is the library under test:

    Linear(C, H, bias=False) -> LIF -> Linear(H, H, bias=False) -> LIF
                             -> Linear(H, 20, bias=False) -> LI readout
    Adam(5e-4), cross-entropy on the time-integrated readout with 0.3 label smoothing, fp32.

Three deliberate concessions, all in the competitors' favour:

* The LI readout is evaluated in closed form. ``sum_t v[t]`` is linear in the readout input,
  so the recurrence collapses to a weighted sum over time -- exact, and it spares the PyTorch
  backends a 256-iteration Python loop that JAX would have fused away.
* SpikingJelly runs multi-step with the CuPy backend, its fastest published path.
* Their decay parameterizations are kept native rather than forced to match: snnTorch learns a
  per-neuron ``beta``, SpikingJelly a shared ``tau``, Norse a fixed one. The arithmetic per
  timestep is the same in each case; only the parameter count of the decay differs.

    python benchmarks/torch_shd.py --library spikingjelly --epochs 20 --trials 3
"""

from __future__ import annotations

import argparse
import os
import time

# jaxpike's SHD loader is pure numpy, but importing it initializes JAX. Keep JAX off the GPU
# so it cannot take memory or a CUDA context away from the library being measured.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
import torch.nn as nn

from benchmarks.shd_data import N_CLASSES, fake, load

# spikingjelly 0.0.0.0.14's CuPy backend still uses `np.int`, removed in NumPy 1.24. Restoring
# the alias is what lets their fused kernels load at all; master has since moved to Triton and
# needs a toolchain this image cannot provide, so the release is the version under test.
if not hasattr(np, "int"):
    setattr(np, "int", int)  # noqa: B010

THRESHOLD = 1.0
SMOOTHING = 0.3


def init_beta(hidden: int, generator: torch.Generator) -> torch.Tensor:
    """Spyx's decay init, reused everywhere: truncated_normal(sd=0.25) + 0.5."""
    beta = torch.empty(hidden)
    nn.init.trunc_normal_(beta, mean=0.5, std=0.25, a=0.0, b=1.0, generator=generator)
    return beta


class LIReadout(nn.Module):
    """``v[t] = beta*v[t-1] + x[t]`` integrated over time, in closed form.

    ``sum_t v[t] = sum_s x[s] * (1 - beta**(T-s)) / (1 - beta)``, which is what the loss
    consumes, so the readout never needs to be unrolled.
    """

    def __init__(self, units: int, generator: torch.Generator):
        super().__init__()
        self.beta = nn.Parameter(init_beta(units, generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (T, B, U) -> (B, U)
        steps = x.shape[0]
        beta = self.beta.clamp(0.0, 0.999)
        exponent = torch.arange(steps, 0, -1, device=x.device, dtype=x.dtype).unsqueeze(1)
        weight = (1.0 - beta.pow(exponent)) / (1.0 - beta)
        return torch.einsum("tbu,tu->bu", x, weight)


class SnnTorchNet(nn.Module):
    def __init__(self, channels: int, hidden: int, generator: torch.Generator):
        super().__init__()
        import snntorch as snn

        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.lif1 = snn.Leaky(
            beta=init_beta(hidden, generator),
            threshold=THRESHOLD,
            learn_beta=True,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.lif2 = snn.Leaky(
            beta=init_beta(hidden, generator),
            threshold=THRESHOLD,
            learn_beta=True,
            reset_mechanism="subtract",
        )
        self.fc3 = nn.Linear(hidden, N_CLASSES, bias=False)
        self.readout = LIReadout(N_CLASSES, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (T, B, C) -> (B, 20)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        current1 = self.fc1(x)
        spikes2 = []
        for step in range(x.shape[0]):
            spike1, mem1 = self.lif1(current1[step], mem1)
            spike2, mem2 = self.lif2(self.fc2(spike1), mem2)
            spikes2.append(spike2)
        return self.readout(self.fc3(torch.stack(spikes2)))


class SpikingJellyNet(nn.Module):
    """Multi-step SpikingJelly with the CuPy backend -- their fastest documented configuration.

    ``v_reset=None`` selects the soft (subtract) reset, and ``decay_input=False`` gives
    ``v[t] = (1 - 1/tau)*v[t-1] + x[t]/tau``: the same decay recurrence with the input scaled,
    which the preceding biasless Linear absorbs.
    """

    def __init__(self, channels: int, hidden: int, generator: torch.Generator, backend: str):
        super().__init__()
        from spikingjelly.activation_based import layer, neuron, surrogate

        def lif():
            return neuron.ParametricLIFNode(
                init_tau=2.0,
                decay_input=False,
                v_threshold=THRESHOLD,
                v_reset=None,
                surrogate_function=surrogate.ATan(alpha=2.0),
                step_mode="m",
                backend=backend,
            )

        self.net = nn.Sequential(
            layer.Linear(channels, hidden, bias=False, step_mode="m"),
            lif(),
            layer.Linear(hidden, hidden, bias=False, step_mode="m"),
            lif(),
            layer.Linear(hidden, N_CLASSES, bias=False, step_mode="m"),
        )
        self.readout = LIReadout(N_CLASSES, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from spikingjelly.activation_based import functional

        functional.reset_net(self.net)
        return self.readout(self.net(x))

    def check_backend(self, requested: str) -> None:
        """SpikingJelly falls back to the Torch backend silently; a fallback is not a fair run."""
        from spikingjelly.activation_based import neuron

        active = {n.backend for n in self.net if isinstance(n, neuron.BaseNode)}
        if active != {requested}:
            raise RuntimeError(f"requested {requested} backend, neurons report {active}")


class NorseNet(nn.Module):
    """Norse's `LIFBoxCell` in a `SequentialState`, its own recommended composition."""

    def __init__(self, channels: int, hidden: int, generator: torch.Generator):
        super().__init__()
        from norse.torch.module.lif import LIFCell

        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.lif1 = LIFCell()
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.lif2 = LIFCell()
        self.fc3 = nn.Linear(hidden, N_CLASSES, bias=False)
        self.readout = LIReadout(N_CLASSES, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state1 = state2 = None
        current1 = self.fc1(x)
        spikes2 = []
        for step in range(x.shape[0]):
            spike1, state1 = self.lif1(current1[step], state1)
            spike2, state2 = self.lif2(self.fc2(spike1), state2)
            spikes2.append(spike2)
        return self.readout(self.fc3(torch.stack(spikes2)))


def build(library: str, channels: int, hidden: int, seed: int, backend: str) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if library == "snntorch":
        return SnnTorchNet(channels, hidden, generator)
    if library == "spikingjelly":
        return SpikingJellyNet(channels, hidden, generator, backend)
    if library == "norse":
        return NorseNet(channels, hidden, generator)
    raise ValueError(f"unknown library {library!r}")


def train(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> float:
    """Whole dataset resident on the GPU, matching how the JAX runs stage theirs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=SMOOTHING)
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    n_batches = len(labels) // batch_size
    usable = n_batches * batch_size

    loss = torch.zeros((), device=inputs.device)
    for _ in range(epochs):
        order = torch.randperm(len(labels), device=inputs.device, generator=generator)[:usable]
        for batch in order.split(batch_size):
            # (B, T, C) uint8 -> (T, B, C) float32, the layout every model here expects.
            xs = inputs[batch].permute(1, 0, 2).float()
            loss = loss_fn(model(xs), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def accuracy(
    model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> float:
    correct = total = 0
    for start in range(0, len(labels) - batch_size + 1, batch_size):
        window = slice(start, start + batch_size)
        xs = inputs[window].permute(1, 0, 2).float()
        correct += int((model(xs).argmax(-1) == labels[window]).sum())
        total += batch_size
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", choices=("snntorch", "spikingjelly", "norse"), required=True)
    ap.add_argument("--backend", default="cupy", help="SpikingJelly neuron backend")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--trials", type=int, default=0, help="timed trials after one warm-up")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/shd")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, tiny")
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}")

    if args.smoke:
        x_train, y_train = fake(512, args.timesteps, args.channels, seed=0)
        x_test, y_test = fake(256, args.timesteps, args.channels, seed=1)
    else:
        x_train, y_train = load(
            "train", args.data, timesteps=args.timesteps, channels=args.channels
        )
        x_test, y_test = load("test", args.data, timesteps=args.timesteps, channels=args.channels)
    print(f"train {x_train.shape} {x_train.dtype}  density {x_train.mean():.4f}")

    train_inputs = torch.from_numpy(x_train).to(device)
    train_labels = torch.from_numpy(y_train.astype(np.int64)).to(device)
    test_inputs = torch.from_numpy(x_test).to(device)
    test_labels = torch.from_numpy(y_test.astype(np.int64)).to(device)

    def fresh() -> nn.Module:
        return build(args.library, args.channels, args.hidden, args.seed, args.backend).to(device)

    label = args.library + (f":{args.backend}" if args.library == "spikingjelly" else "")
    print(f"library {label}  hidden {args.hidden}  batch {args.batch}  T {args.timesteps}")

    probe = fresh()
    if isinstance(probe, SpikingJellyNet):
        probe.check_backend(args.backend)
    print(f"parameters: {sum(p.numel() for p in probe.parameters())}")

    if not args.skip_accuracy:
        model = fresh()
        loss = train(
            model,
            train_inputs,
            train_labels,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=args.seed,
        )
        model.eval()
        test_acc = accuracy(model, test_inputs, test_labels, args.batch)
        print(f"final train loss {loss:.4f}   TEST ACCURACY: {test_acc:.4f}")

    if args.trials:
        on_cuda = device.type == "cuda"
        times = []
        for trial in range(args.trials + 1):
            if on_cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            train(
                fresh(),
                train_inputs,
                train_labels,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                seed=args.seed,
            )
            if on_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(
                f"  trial {trial}{' (warm-up, discarded)' if trial == 0 else ''}: {elapsed:.2f} s"
            )
        timed = np.array(times[1:])
        print(f"TIME {args.epochs} epochs: {timed.mean():.2f} +/- {timed.std():.2f} s")
        if on_cuda:
            print(f"peak allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
