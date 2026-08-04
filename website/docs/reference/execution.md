---
id: execution
title: Execution API
sidebar_position: 4
---

# Execution API

Conceptual treatment and the measured numbers are in
[Execution and parallel-in-time](../guides/execution.md); this page is the signatures.

## `unroll`

```python
jp.unroll(net, xs, state=None) -> (outputs, final_state)
```

Runs `net` over the leading (time) axis of `xs`, which is `(T, batch, ...)`. Returns the
per-timestep outputs stacked on the leading axis plus the final state, so a long sequence can be
processed in chunks by feeding the returned state back in. If `state` is `None`, it calls
`net.init_state(xs.shape[1:])`.

One `lax.scan` step per timestep, with membrane state materialized at every step so BPTT can
reach it: `O(T·B·N)` memory. This is the reference path and supports every neuron model.

## `unroll_checkpointed`

```python
jp.unroll_checkpointed(net, xs, state=None, chunk_size=None) -> (outputs, final_state)
```

Identical results to `unroll`; the backward pass re-runs each chunk's forward instead of keeping
every timestep's residuals live. Peak scratch drops from `O(T·B·N)` to roughly `O(sqrt(T)·B·N)`
for one extra forward pass — 67× less memory at `T=5000` for a 1.09× time cost on a T4.

`chunk_size` defaults to the divisor of `T` closest to `sqrt(T)`, and **must divide `T`**.
Padding is refused rather than supported: padded timesteps would still advance the recurrence
and silently corrupt the returned final state.

## `unroll_parallel`

```python
jp.unroll_parallel(net, xs, state=None) -> (outputs, final_state)
```

Solves the whole time axis with an associative scan over the affine recurrence. Requires every
layer to be stateless or reset-free; raises by name for anything else, including recurrent
`Graph`s.

Results are bit-identical to `unroll` in the spike train (membrane values differ by at most
6e-07 from accumulation order). Costs memory: it materializes the full `[T, B, N]` activation
tensor.

```python
jp.parallel.supports_parallel(layer) -> bool
```

Whether a single layer can take the parallel path — true if it implements `parallel_apply`.

```python
jp.parallel.scan_linear_recurrence(a, b, v0) -> v
```

Solves `v[t] = a[t]*v[t-1] + b[t]` for all `t` with an associative scan. This is the primitive
to use when adding `parallel_apply` to your own neuron.

## Readouts

```python
jp.spike_rate(spikes)              # mean over time
jp.spike_count(spikes)             # sum over time
jp.density(spikes)                 # fraction of (timestep, neuron) slots that spiked
jp.max_membrane_logits(membrane)   # max over time -- pair with a LeakyIntegrator readout
jp.count_logits(spikes)            # alias for the spike-count readout
```

`density` is worth watching during development: it catches a layer that has gone silent, and
below roughly 10% a sparse gather-and-accumulate path would beat a dense matmul.
