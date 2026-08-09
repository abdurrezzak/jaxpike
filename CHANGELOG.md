# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Layer-wise execution: `unroll` and `unroll_checkpointed` hoist stateless layers out of the
  time loop and evaluate them once across all timesteps.
- `scan_unroll` on both sequential paths, controlling how many timesteps are emitted per loop
  iteration. Defaults to 8, chosen by measurement.
- `remat_step` on `unroll`, recomputing the neuron step in the backward pass instead of
  storing its residuals. Trades ~9% speed for ~22% less peak scratch, with bit-identical
  gradients.
- Comparative benchmark suite against snnTorch, SpikingJelly, Norse and Spyx, with every
  framework run in one container on one GPU.
- Automated optimization search that gates candidates on agreement with the reference in both
  outputs and gradients before timing them.

### Changed

- Surrogate gradients use `custom_jvp` rather than the straight-through identity, so the
  relaxation is no longer evaluated in the forward pass. Subclasses are unaffected: they still
  define only `relaxation`.

## [0.0.1] — unreleased

Initial development version: neuron models, arbitrary topologies, three execution strategies,
surrogate gradients, plasticity rules, NIR interchange, visualization, and e-prop.
