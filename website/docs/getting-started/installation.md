---
id: installation
title: Installation
sidebar_position: 1
---

# Installation

jaxpike is not published to PyPI yet. Install it from a checkout:

```bash
git clone <repository-url> jaxpike
cd jaxpike
uv venv && uv pip install -e ".[dev]"
```

`pip install -e ".[dev]"` works identically if you don't use `uv`.

## Optional extras

Each extra is independent, and the features that need one degrade with a clear error rather
than a missing-import traceback. Tests for an absent extra skip.

| Extra | Pulls in | Needed for |
|---|---|---|
| `[viz]` | matplotlib | everything in `jaxpike.viz` |
| `[data]` | h5py | `jaxpike.data.shd`, `jaxpike.data.ssc` |
| `[nir]` | nir | `jaxpike.nir` import and export |
| `[bench]` | torch, snnTorch, Norse, SpikingJelly | the cross-framework comparison tests |
| `[dev]` | pytest, ruff | development |

```bash
uv pip install -e ".[dev,viz,data]"    # the usual working set
```

## Verify the install

```bash
.venv/bin/pytest                                        # ~2 min, CPU only
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## GPUs

The library itself is pure JAX and runs on whatever backend JAX has. Two caveats:

**macOS cannot run the kernel or benchmark work.** JAX's Metal backend has no Triton or Mosaic
support, so Pallas does not run there at all. Benchmarks in this repository go through
[Modal](https://modal.com):

```bash
.venv/bin/python -m modal setup                                    # once
.venv/bin/python -m modal run benchmarks/gpu/run_modal.py --bench memory
.venv/bin/python -m modal run benchmarks/gpu/run_modal.py --bench parallel
.venv/bin/python -m modal run benchmarks/gpu/run_modal.py --bench network
.venv/bin/python -m modal run benchmarks/gpu/run_modal.py --bench shd \
    --extra "--epochs 100 --recurrent --augment"
```

Add `--gpu A100` if your Modal account has billing configured; the free tier is T4-only, and
every published number here is a T4 number.

**CUDA JAX is a separate wheel.** Install `jax[cuda12]` following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for your CUDA
version; jaxpike does not pin it.
