# jaxpike

**Spiking neural networks in JAX.** Fast, functional, and honest about its numbers.

[![CI](https://github.com/abdurrezzak/jaxpike/actions/workflows/ci.yml/badge.svg)](https://github.com/abdurrezzak/jaxpike/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

```python
import jax
import jaxpike as jp

key = jax.random.key(0)
k1, k2 = jax.random.split(key)

net = jp.Sequential(
    jp.Dense(784, 512, key=k1),
    jp.LIF(tau=20.0),
    jp.Dense(512, 10, key=k2),
    jp.LIF(tau=20.0, surrogate=jp.ATan()),
)

xs = jax.random.uniform(key, (100, 32, 784))   # (time, batch, features)
spikes, state = jp.unroll(net, xs)
logits = jp.spike_rate(spikes)
```

---

## Why jaxpike

Spiking networks are recurrent networks with a non-differentiable activation, and the
frameworks that train them force a choice. Hand-written CUDA kernels are fast but only support
the neuron models someone already wrote a kernel for. Pure PyTorch and JAX implementations let
you write any neuron and run an order of magnitude slower.

jaxpike takes the second path and closes most of the gap by construction rather than by hand:

- **Layer-wise execution.** Stateless layers are hoisted out of the time loop and evaluated
  once across all timesteps, so a `Dense` becomes one large matrix multiply instead of `T`
  small ones.
- **Three execution strategies, one signature.** `unroll` for sequential BPTT,
  `unroll_checkpointed` for `O(√T)` memory, `unroll_parallel` for `O(log T)` depth on
  reset-free neurons. They are interchangeable and produce the same answers.
- **Surrogate gradients without custom VJPs.** Write the smooth relaxation; the derivative
  comes from autodiff, so forward and backward cannot disagree.
- **Explicit functional state.** Long sequences stream in chunks and truncated BPTT is free.
- **NIR import and export**, so trained models move to Loihi, SpiNNaker2, Speck and the rest
  of the neuromorphic ecosystem.

## Installation

```bash
pip install jaxpike
```

For a GPU, install JAX for your platform first — see the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html):

```bash
pip install -U "jax[cuda12]"
pip install jaxpike
```

From a checkout, with the development and benchmark extras:

```bash
uv venv && uv pip install -e ".[dev]"
```

Requires Python 3.11+.

## Documentation

Full documentation, tutorials and API reference live in [`website/`](website/) and cover the
[quickstart](website/docs/getting-started/quickstart.md), a
[worked SHD training run](website/docs/getting-started/training-shd.md), a
[migration guide from snnTorch](website/docs/guides/coming-from-snntorch.md), and
[why deep SNNs go silent](website/docs/guides/silent-networks.md).

```bash
cd website && npm install && npm start   # Node 20+
```

## Benchmarks

Every framework installed side by side and trained on identical arrays with the same model,
optimizer, loss and dtype, in one container on one NVIDIA T4. SHD, hidden 128, 20 epochs at
batch 256, T=256. Full protocol, ablations and unfavourable results in
[`benchmarks/README.md`](benchmarks/README.md).

| framework | training time | peak memory |
|---|---:|---:|
| SpikingJelly 0.0.0.0.14, multi-step + CuPy | **6.02 s** | 792.1 MB |
| **jaxpike, `unroll`** | **8.12 s** | 324.5 MB |
| **jaxpike, `unroll_checkpointed`** | 11.07 s | **64.2 MB** |
| jaxpike, `unroll_parallel` | 15.80 s | 288.1 MB |
| Norse 1.1.0 | 252.21 s | 737.3 MB |
| SpikingJelly, Torch backend | 260.62 s | 696.3 MB |
| snnTorch 1.0.0 | 347.18 s | 675.8 MB |

Accuracy is matched rather than traded away: **0.751 on SHD** under Spyx's published protocol,
against their reported 0.70–0.75 band.

Anything that steps through time in a Python loop is 31–43× slower, which is most of the
field. `unroll_checkpointed` holds a 256-step BPTT graph in 64 MB where SpikingJelly needs
792 MB. SpikingJelly's fused CuPy kernel remains **1.35× faster** than the best jaxpike path
and is not beaten; the gap is entirely in the neuron time loop, where 83% of a training step
is spent.

## Core concepts

### Defining a neuron

Any module following the state contract works. Nothing is registered, subclassed or
special-cased:

```python
init_state(input_shape) -> state pytree
out_shape(input_shape)  -> output shape
__call__(state, x)      -> (new_state, spikes)
```

Add an optional `parallel_apply(state, xs)` and the layer becomes eligible for
`unroll_parallel`.

### Defining a surrogate gradient

Write the smooth relaxation. The forward pass emits an exact binary spike and the backward
pass differentiates the relaxation, so the two cannot drift apart, and the derivative can be
finite-difference tested:

```python
class MySurrogate(jp.Surrogate):
    slope: float = 10.0

    def relaxation(self, v):
        return jax.nn.sigmoid(self.slope * v)
```

### Arbitrary topologies

`Sequential` is a straight chain. `Graph` wires any layer to any other — recurrence, skip
connections, branching, fan-in:

```python
net = jp.Graph(
    nodes={
        "w_in": jp.Dense(700, 128, key=k1),
        "hidden": jp.LIF(tau=20.0),
        "w_rec": jp.Dense(128, 128, key=k2),
        "w_out": jp.Dense(128, 20, key=k3),
        "out": jp.LeakyIntegrator(tau=20.0),
    },
    edges=[
        ("input", "w_in"),
        ("w_in", "hidden"),
        ("hidden", "w_rec"),
        ("w_rec", "hidden"),      # closes a cycle
        ("hidden", "w_out"),
        ("w_out", "out"),
    ],
    output="out",
)
```

![Architecture diagrams](docs/figures/architecture_light.png)

Two rules make any wiring well-defined. **A node with several incoming edges sums them**,
which is what a synapse does and what makes fan-in and skip connections work without special
syntax. **An edge that closes a cycle reads the previous timestep**, because a cycle cannot be
resolved within one step — that is what makes a recurrent SNN recurrent, and `Graph` finds the
back-edges automatically.

A recurrent `Graph` cannot run parallel-in-time and raises rather than quietly computing
something else.

### Spiking convnets

Layout is NHWC — `(time, batch, height, width, channels)` — because XLA's convolutions are
written for channels-last and NCHW forces a transpose around every operation.

```python
gain = jp.lif_gain(tau=20.0)

net = jp.Sequential(
    jp.Conv2d(2, 32, 3, key=k1, gain=gain),   # 2 channels: DVS on/off events
    jp.LinearLIF(tau=20.0, threshold=0.2),
    jp.Pool2d(2),
    jp.Conv2d(32, 64, 3, key=k2, gain=gain),
    jp.LinearLIF(tau=20.0, threshold=0.2),
    jp.Pool2d(2),
    jp.Flatten(),
    jp.Dense(64 * 8 * 8, 10, key=k3, gain=gain),
    jp.LinearLIF(tau=20.0, threshold=0.2),
)
```

Convolution and pooling are stateless, so a spiking convnet runs through `unroll_parallel`
end to end.

### Why deep SNNs go silent

Deep spiking networks have a failure mode that ANNs do not: activity decays multiplicatively
with depth until nothing reaches the output, and a silent network has no gradient anywhere to
recover from. A three-layer spiking convnet with standard LeCun initialization:

| | layer 1 | layer 2 | layer 3 |
|---|---:|---:|---:|
| plain LeCun init | 0.045 | **0.000** | **0.000** |
| with `gain=lif_gain(tau)` | 0.380 | 0.327 | 0.200 |

A LIF membrane is an exponential moving average, which attenuates signal standard deviation by
`sqrt((1-a)/(1+a))` — a factor of 6.3 at `tau=20`. Weights initialized for unit-variance
activations therefore produce membranes six times smaller than intended, sitting below
threshold. `jp.lif_gain(tau)` returns the compensating factor.

## Visualization

![Visualization gallery](docs/figures/gallery_light.png)

```python
from jaxpike import viz

viz.raster(spikes)
viz.membrane(voltages, spikes=spikes, threshold=1.0)
viz.layer_rates_from(net, xs)       # check for silent or saturated layers
viz.rate_heatmap(spikes)
viz.weights(net.layers[0].weight)
```

Every function takes an optional `ax` and returns it, so plots compose into larger figures.
`viz.Theme.dark()` switches to a dark surface with its own selected colour steps rather than an
inverted light palette.

`layer_rates_from` is the one to reach for first: it plots the firing rate after every spiking
layer and labels any that have gone silent or saturated.

## Hardware export via NIR

[NIR](https://github.com/neuromorphs/NIR) is the field's interchange format. Exporting to it
lets a model trained here run in snnTorch, Norse, Spyx, Lava, Rockpool or Nengo, and deploy to
Intel Loihi, SpiNNaker2, BrainScaleS-2, SynSense Speck or Xylo.

```python
from jaxpike import nir

nir.save(net, "model.nir", input_shape=(1, 700), dt_seconds=1e-3)
net = nir.load("model.nir")
```

Three things to know, all verified by round-trip tests:

**Units are not standardized by NIR.** It stores `tau` in seconds; jaxpike stores it in
timesteps. `dt_seconds` declares what one timestep physically means, and getting it wrong
rescales every time constant in the model.

**Some models cannot be exported, and those raise rather than silently changing.**
`reset="subtract"` has no NIR equivalent, and neither do max pooling, adaptive thresholds,
Izhikevich dynamics or short-term plasticity.

**Leaving the library is not bit-exact.** NIR specifies a differential equation, not a
discretization — jaxpike solves it exactly, Norse uses forward Euler. snnTorch assumes
`dt = 1e-4 s` regardless of the file and has no mapping for NIR's `LI` node. Check numerically
on the far side.

## Roadmap

Implemented and benchmarked: the neuron zoo, arbitrary topologies, three execution strategies,
surrogate gradients, plasticity, NIR interchange, visualization, and e-prop.

Not yet built, in priority order:

- **Compiled neuron kernels.** Tracing a user's neuron to a jaxpr and generating a fused
  Pallas kernel plus its custom VJP. This is the project's central bet and is currently
  untested: Pallas requires compute capability 8.0 or higher.
- **Parallel-in-time for reset neurons.** Chunked scan and DEER-style fixed-point iteration,
  extending `unroll_parallel` beyond the reset-free case.
- **Sparse event-driven matmul.** Spikes are 1–5% dense; dense compute discards most of that.
- **More learning rules.** OTTT, SLTT, FPTT and forward-mode alongside BPTT and e-prop.
- **Multi-device** sharding over batch and time.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Benchmarks against other frameworks require an NVIDIA GPU and are run remotely; see
[`benchmarks/README.md`](benchmarks/README.md).

## Citation

```bibtex
@software{jaxpike,
  title  = {jaxpike: spiking neural networks in JAX},
  author = {Efe, Abdurrezak},
  year   = {2026},
  url    = {https://github.com/abdurrezzak/jaxpike}
}
```

## License

Apache-2.0.
