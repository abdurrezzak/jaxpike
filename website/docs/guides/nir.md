---
id: nir
title: Hardware export via NIR
sidebar_position: 9
---

# Hardware export via NIR

[NIR](https://github.com/neuromorphs/NIR) is the field's interchange format — ONNX for spiking
networks. Exporting to it lets a model trained here run in snnTorch, Norse, Spyx, Lava,
Rockpool or Nengo, and deploy to Intel Loihi, SpiNNaker2, BrainScaleS-2, SynSense Speck or Xylo.

Needs the `[nir]` extra.

```python
from jaxpike import nir

nir.save(net, "model.nir", input_shape=(1, 700), dt_seconds=1e-3)
net = nir.load("model.nir")        # exact round trip, including convnets
```

`to_nir(module, input_shape, dt_seconds=...)` and `from_nir(graph, dt=..., dt_seconds=...)` are
the in-memory forms.

## Units are not standardized by NIR

NIR stores `tau` in **seconds**; jaxpike neurons store it in **timesteps**. `dt_seconds` declares
what one of your timesteps physically means, and getting it wrong rescales every time constant
in the model. The default is `1e-3`, i.e. one timestep is one millisecond.

This is the single most common way to get a model that loads cleanly and behaves nothing like
the one you trained.

## Channel ordering: NIR is channels-first, jaxpike is channels-last

Beyond transposing conv weights, this changes the feature *ordering* that a flatten produces,
so the following `Dense` layer needs its columns permuted. Skip that and you get a model with
correct shapes that runs fine and computes something different.

`jaxpike.nir` handles this, and there is a test dedicated to it.

## Some models cannot be exported, and those raise

Rather than silently changing your model, export raises `NIRConversionError` for:

- `LIF(reset="subtract")` — NIR's LIF resets to a fixed `v_reset` and cannot express
  subtract-reset. **This is the jaxpike default**, so a portable model must be built with
  `reset="zero"`.
- max pooling (`Pool2d(mode="max")`) — NIR has no max-pooling primitive
- adaptive thresholds (`ALIF`)
- `Izhikevich` dynamics
- short-term plasticity (`TsodyksMarkram`)

`LinearLIF` exports as an `LI` node feeding a `Threshold` node. NIR has no single primitive for
a reset-free spiking neuron, but the two-node form is exact rather than an approximation.

## Leaving the library is not bit-exact

NIR specifies a differential equation, not a discretization. jaxpike solves it exactly in
closed form; Norse uses forward Euler, which carries `O(dt/tau)` truncation error. Both are
correct implementations of the same model and they agree in the limit as `dt` shrinks — the
cross-framework tests measure exactly that — but they will not match step for step.

Two snnTorch-specific caveats, verified rather than assumed:

- snnTorch's importer assumes `dt = 1e-4 s` regardless of what the file says.
- It has no mapping for NIR's `LI` node, so a `LeakyIntegrator` readout will not cross into it.

**Check numerically on the far side.** Round-tripping within jaxpike is exact; crossing to
another framework is not, and the discrepancy is a property of the format rather than a bug in
either implementation.
