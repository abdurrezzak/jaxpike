# Contributing

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
```

## Before opening a pull request

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Doctests run as part of the suite, so examples in docstrings are executed rather than assumed.

## Standards

**New behaviour needs a test that fails without it.** The test suite is the specification;
several of this library's subtler properties — that `unroll_checkpointed` is indistinguishable
from `unroll`, that layer hoisting does not change gradients — exist only because a test
asserts them.

**Performance changes need a measurement.** `benchmarks/autosearch.py` gates a candidate on
agreeing with the reference in both outputs and gradients before timing it, and reports the
reference measured against itself as a noise floor. A change inside that floor is a tie, not an
improvement.

**Numerical changes must be stated.** If an optimization reassociates floating-point arithmetic
it is no longer bit-identical, and the documentation must say so with a magnitude rather than
claiming equivalence.

**Documentation examples must run.** Every code block in the docs site should execute against
the current API. Signatures written from memory tend to be wrong.

## Adding a neuron

A neuron is any module satisfying the state contract. Nothing is registered or subclassed:

```python
init_state(input_shape) -> state pytree
out_shape(input_shape)  -> output shape
__call__(state, x)      -> (new_state, spikes)
```

Add `parallel_apply(state, xs)` if the recurrence is linear in time, which makes the layer
eligible for `unroll_parallel`. Reset makes a neuron nonlinear, so reset-based models must not
define it.

## Adding a surrogate gradient

Subclass `Surrogate` and implement `relaxation`. The forward pass emits an exact binary spike
and the backward pass differentiates the relaxation, so no custom VJP is involved and the
gradient can be finite-difference tested.

## Benchmarks

Benchmarks that compare against other frameworks require an NVIDIA GPU and run remotely
through Modal. See [`benchmarks/README.md`](benchmarks/README.md). Results are committed with
unfavourable outcomes intact.
