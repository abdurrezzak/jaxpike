# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] — 2026-08-10

### Documentation

- The README, which is the project description on PyPI, never mentioned the plasticity rules.
  `STDP`, `stdp_window`, `TsodyksMarkram` with the five `MARKRAM_PRESETS`, and `DopamineSTDP`
  have shipped since 0.1.0 but were invisible to anyone reading the front page. Added a section
  with worked examples.

## [0.1.1] — 2026-08-10

### Fixed

- `Homepage` and `Documentation` pointed at the repository rather than the documentation site,
  so the Documentation link on PyPI led to GitHub. Both now point to
  https://abdurrezzak.github.io/jaxpike/.

## [0.1.0] — 2026-08-09

First public release.

### Neurons and layers

- `LIF`, `LinearLIF`, `ALIF`, `Izhikevich` with seven firing-pattern presets, and
  `LeakyIntegrator` as a readout.
- `Dense`, `Conv2d`, `Pool2d`, `Flatten`, and the `Sequential` container.
- `Graph` for arbitrary topologies: recurrence, skip connections, branching and fan-in, with
  back-edges resolved automatically.
- `lif_gain` for initialization that keeps deep spiking networks from going silent.

### Execution

- `unroll` for sequential BPTT, `unroll_checkpointed` for `O(√T)` memory, and
  `unroll_parallel` for `O(log T)` depth on reset-free neurons. All three share a signature and
  agree to float32 noise.
- Stateless layers are hoisted out of the time loop and evaluated once across all timesteps, so
  a `Dense` compiles to one large matrix multiply rather than `T` small ones.
- `scan_unroll` controls how many timesteps are emitted per loop iteration; the default of 8
  was chosen by measurement.
- `remat_step` recomputes the neuron step in the backward pass rather than storing residuals,
  trading roughly 9% speed for 22% less peak scratch with bit-identical gradients.

### Gradients and learning

- Surrogate gradients defined by a smooth relaxation, with the derivative obtained from
  autodiff via `custom_jvp` — no hand-written VJP, and the relaxation is not evaluated in the
  forward pass. `FastSigmoid`, `ATan`, `Triangular` and `Boxcar` are included.
- e-prop for online learning with memory flat in `T`.
- `STDP`, `TsodyksMarkram` short-term plasticity with Markram presets, and reward-modulated
  `DopamineSTDP`.

### Ecosystem

- NIR import and export, round-trip tested, with unrepresentable models raising rather than
  silently changing.
- SHD and SSC dataset loaders.
- Visualization: raster plots, membrane traces, firing-rate diagnostics, weight matrices and
  architecture diagrams, with light and dark themes.

### Benchmarks

Measured against snnTorch, SpikingJelly, Norse and Spyx, every framework running the identical
model on identical arrays in one container on one NVIDIA T4. jaxpike trains 31–43× faster than
snnTorch and Norse at 12× less memory than SpikingJelly, and remains 1.35× slower than
SpikingJelly's fused CuPy kernel.

Accuracy on SHD is 0.7532 ± 0.0292 over five seeds, at the upper edge of the 0.70–0.75 band
published for Spyx under the same protocol. Full protocol, ablations and negative results in
`benchmarks/README.md`.
