# jaxpike

Fast, flexible spiking neural networks in JAX.

> **Status: pre-alpha.** The reference implementation, test suite, and the first fast path
> are in place. Measured on an NVIDIA T4: **21.7× faster training at T=8192** for a
> feedforward network using parallel-in-time execution, with bit-identical spike trains, and
> **67× lower BPTT memory** via rematerialization. Both reproducible from
> [`benchmarks/`](benchmarks/). Parallel-in-time currently requires reset-free neurons
> (`LinearLIF`); the general reset case is not solved yet. See [PLAN.md](PLAN.md).

## Why another SNN library

The neuromorphic community frames this field as one tradeoff: **speed versus flexibility.**
Frameworks with hand-written CUDA kernels are the fastest available but only support the
neuron models someone already wrote a kernel for. Frameworks in pure PyTorch or JAX let you
write any neuron you like and run considerably slower.

jaxpike exists because that tradeoff is an artifact of kernels being written *by hand*. If a
neuron model is **compiled** into a fused kernel instead, you get both. That is the
project's central bet, and everything else follows from it:

- **Compiled neurons.** Write a neuron as an ordinary JAX function; get a fused kernel and
  its custom VJP generated for you, with a clear diagnostic and a correct fallback when your
  model falls outside the compilable subset.
- **Memory that doesn't scale with time.** Every framework today stores membrane state for
  all `T` timesteps to support BPTT, at `O(T·B·N)`. Fusing the recurrence and recomputing in
  the backward pass makes the working set `O(B·N)`.
- **Leaping over time, not stepping through it.** Named for saltatory conduction: myelinated
  axons don't regenerate the action potential at every point along the membrane, which is why
  they conduct at 150 m/s instead of 10. Same idea — parallel scan within a chunk, state
  regenerated only at checkpoints.
- **Learning rules as a first-class axis.** BPTT is the default, not the only option. e-prop,
  OTTT, SLTT and forward-mode alternatives against the same model definition.
- **Portable.** One kernel source targeting both NVIDIA GPUs and TPUs, via Pallas.
- **Interoperable.** NIR import and export, so models move between jaxpike, Norse, snnTorch,
  Loihi, SpiNNaker2 and Speck.

## Install

```bash
pip install -e ".[dev]"     # from a checkout; not yet on PyPI
```

## Quick start

```python
import jax
import jax.numpy as jnp
import jaxpike as jp

key = jax.random.key(0)
k1, k2 = jax.random.split(key)

net = jp.Sequential(
    jp.Dense(784, 512, key=k1),
    jp.LIF(tau=20.0),  # tau is learnable by default
    jp.Dense(512, 10, key=k2),
    jp.LIF(tau=20.0, surrogate=jp.ATan()),
)

xs = jax.random.uniform(key, (100, 32, 784))  # (time, batch, features)
spikes, final_state = jp.unroll(net, xs)
logits = jp.spike_rate(spikes)  # rate-coded readout
```

State is explicit and functional, so long sequences stream in chunks and truncated BPTT
comes for free:

```python
spikes_a, state = jp.unroll(net, xs[:50])
spikes_b, state = jp.unroll(net, xs[50:], state)  # exactly equals the unchunked run
```

## Defining a neuron

Any module following the state contract works. Nothing is registered, subclassed, or
special-cased:

```python
init_state(input_shape) -> state pytree
out_shape(input_shape)  -> output shape
__call__(state, x)      -> (new_state, spikes)
```

## Defining a surrogate gradient

Write the smooth relaxation; the gradient comes from autodiff. There is no custom VJP to get
wrong, and because the derivative is autodiff-consistent it can be finite-difference tested:

```python
class MySurrogate(jp.Surrogate):
    slope: float = 10.0

    def relaxation(self, v):
        return jax.nn.sigmoid(self.slope * v)
```

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest              # CPU, fast
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Kernel and benchmark work requires NVIDIA or TPU hardware — JAX's Metal backend cannot run
Pallas, so those are remote-only. See §3 of [PLAN.md](PLAN.md).

## License

Apache-2.0.
