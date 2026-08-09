"""Automated optimization search over execution strategies.

Each candidate is a different way of running the same network, and is admitted only if it
survives four rules:

1. **Correctness gates timing.** A candidate is compared against the reference on outputs
   *and* gradients and rejected if it disagrees by more than its declared tolerance. Whether
   it was bit-identical is reported separately: reassociation inside a fused loop body
   perturbs results at around 1e-9, which is harmless but is not zero.
2. **Each candidate is measured in its own process**, against a reference timed on either side
   of it. Sharing one interpreter lets held-open arrays and accumulated models inflate later
   timings by ~30%, and a paired reference cannot correct for a device that degraded under
   both.
3. **Compile and steady state are reported separately.** They are paid on different schedules
   and only one is what a training run spends its time on.
4. **The reference is timed against itself** and the result published as a noise floor.
   Candidates inside it are reported as ties rather than ranked.

Results are written as JSON so that a later run compares against an earlier one.

    PYTHONPATH=. python benchmarks/autosearch.py --out results/autosearch.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jaxpike as jp
from benchmarks._measure import compile_stats
from benchmarks.spyx_shd import build, integral_crossentropy

REFERENCE = "sequential"


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One way of running the network, and how closely it must match the reference."""

    name: str
    runner: Callable
    neuron: str = "lif"
    tolerance: float = 1e-5
    note: str = ""
    forward_only: bool = False


@dataclasses.dataclass
class Result:
    name: str
    note: str
    ok: bool
    exact: bool | None = None
    output_error: float | None = None
    gradient_error: float | None = None
    compile_seconds: float | None = None
    step_ms: float | None = None
    reference_ms: float | None = None
    speedup: float | None = None
    scratch_mb: float | None = None
    error: str | None = None


class Passthrough(eqx.Module):
    """Stands in for a layer so its cost can be subtracted from the total."""

    stateful: bool = eqx.field(static=True)

    def init_state(self, input_shape):
        return jnp.zeros(input_shape) if self.stateful else None

    def out_shape(self, input_shape):
        return input_shape

    def __call__(self, state, x):
        return (x if self.stateful else None), x

    def parallel_apply(self, state, xs):
        return (xs[-1] if self.stateful else None), xs


def ablate(model, *, drop: str):
    """The same network with one class of layer replaced by a passthrough.

    Timing a step says how long it takes, not where the time went. Subtracting one class of
    layer at a time localizes the cost.
    """
    layers = []
    for layer in model.layers:
        is_dense = isinstance(layer, jp.Dense)
        # A Dense that changes width cannot be dropped without breaking every layer after it,
        # so the readout projection stays in both ablations.
        square = is_dense and layer.weight.shape[0] == layer.weight.shape[1]
        if drop == "gemms" and square:
            layers.append(Passthrough(stateful=False))
        elif drop == "neurons" and not is_dense:
            layers.append(Passthrough(stateful=True))
        else:
            layers.append(layer)
    return jp.Sequential(*layers)


def candidates(timesteps: int) -> list[Candidate]:
    """The search space. Extend here; the loop needs no other change."""
    space = [
        Candidate(REFERENCE, jp.unroll, note="reference"),
        Candidate(
            "no-neurons",
            lambda m, xs, **kw: jp.unroll(ablate(m, drop="neurons"), xs, **kw),
            tolerance=float("inf"),
            note="diagnostic: GEMMs only",
        ),
        Candidate(
            "no-gemms",
            lambda m, xs, **kw: jp.unroll(ablate(m, drop="gemms"), xs, **kw),
            tolerance=float("inf"),
            note="diagnostic: neuron loops only",
        ),
        Candidate(
            "forward-only",
            jp.unroll,
            tolerance=float("inf"),
            forward_only=True,
            note="diagnostic: no backward pass",
        ),
        Candidate(
            "parallel",
            jp.unroll_parallel,
            neuron="psu",
            tolerance=float("inf"),
            note="reset-free neuron, different model",
        ),
    ]
    for factor in (1, 2, 4, 8, 16, 32):
        space.append(
            Candidate(
                f"unroll-{factor}",
                lambda m, xs, f=factor: jp.unroll(m, xs, scan_unroll=f),
                note=f"{factor} timesteps per loop iteration",
            )
        )
    for factor in (1, 4, 8, 16):
        space.append(
            Candidate(
                f"remat-step-{factor}",
                lambda m, xs, f=factor: jp.unroll(m, xs, scan_unroll=f, remat_step=True),
                note=f"recompute the step in backward, {factor} per iteration",
            )
        )
    for chunk in sorted({c for c in (8, 16, 32, 64) if timesteps % c == 0}):
        space.append(
            Candidate(
                f"checkpointed-{chunk}",
                lambda m, xs, c=chunk: jp.unroll_checkpointed(m, xs, chunk_size=c),
                note=f"rematerialized in chunks of {chunk}",
            )
        )
    return space


def make_step(model, runner, optimizer, *, forward_only: bool = False):
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(p, xs, labels):
        traces, _ = runner(eqx.combine(p, static), xs)
        return integral_crossentropy(traces, labels)

    def step(p, opt_state, xs, labels):
        if forward_only:
            return p, opt_state, loss_fn(p, xs, labels)
        loss, grads = jax.value_and_grad(loss_fn)(p, xs, labels)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return eqx.apply_updates(p, updates), opt_state, loss

    return params, static, loss_fn, step


def reference_gradients(model, xs, labels):
    params, _, loss_fn, _ = make_step(model, jp.unroll, optax.adam(5e-4))
    outputs, _ = jp.unroll(model, xs)
    return outputs, jax.grad(loss_fn)(params, xs, labels)


def verify(candidate: Candidate, model, xs, labels, reference) -> tuple[float, float]:
    """Largest absolute disagreement with the reference, in outputs and in gradients."""
    ref_outputs, ref_grads = reference
    params, _, loss_fn, _ = make_step(model, candidate.runner, optax.adam(5e-4))
    outputs, _ = candidate.runner(model, xs)
    grads = jax.grad(loss_fn)(params, xs, labels)

    output_error = float(jnp.max(jnp.abs(outputs - ref_outputs)))
    gradient_error = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(grads), jax.tree.leaves(ref_grads), strict=True)
    )
    return output_error, gradient_error


WARM_UP_SECONDS = 2.0


def measure(candidate: Candidate, model, xs, labels, repeats: int) -> tuple[float, float, float]:
    optimizer = optax.adam(5e-4)
    params, _, _, step = make_step(
        model, candidate.runner, optimizer, forward_only=candidate.forward_only
    )
    opt_state = optimizer.init(params)
    compiled = eqx.filter_jit(step)

    start = time.perf_counter()
    out = compiled(params, opt_state, xs, labels)
    jax.block_until_ready(out)
    compile_seconds = time.perf_counter() - start

    # A GPU idles at a low clock and needs sustained load to boost; timing from cold
    # measures the clock ramp rather than the kernel.
    deadline = time.perf_counter() + WARM_UP_SECONDS
    while time.perf_counter() < deadline:
        jax.block_until_ready(compiled(params, opt_state, xs, labels))

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = compiled(params, opt_state, xs, labels)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)

    scratch = compile_stats(step, params, opt_state, xs, labels)["temp_bytes"]
    # The minimum is the sample least contaminated by scheduling and thermal interference.
    return compile_seconds, float(np.min(times)) * 1e3, scratch / 2**20


def search(args) -> list[Result]:
    key = jax.random.key(0)
    xs = (jax.random.uniform(key, (args.timesteps, args.batch, args.channels)) < 0.05).astype(
        jnp.float32
    )
    labels = jax.random.randint(key, (args.batch,), 0, 20)

    def model_for(neuron: str):
        return build(args.hidden, args.channels, key, neuron_kind=neuron)

    baseline = next(c for c in candidates(args.timesteps) if c.name == REFERENCE)
    candidate = next(c for c in candidates(args.timesteps) if c.name == args.only)
    model = model_for(candidate.neuron)

    result = Result(candidate.name, candidate.note, ok=True)
    if candidate.tolerance != float("inf"):
        reference = reference_gradients(model_for("lif"), xs, labels)
        output_error, gradient_error = verify(candidate, model, xs, labels, reference)
        del reference
        result.output_error = output_error
        result.gradient_error = gradient_error
        worst = max(output_error, gradient_error)
        result.exact = worst == 0.0
        if worst > candidate.tolerance:
            result.ok = False
            result.error = f"disagrees with reference by {worst:.3e}"

    # Timed on either side of the candidate so the ratio reflects the device as it was.
    reference_model = model_for(baseline.neuron)
    _, before, _ = measure(baseline, reference_model, xs, labels, args.repeats)
    result.compile_seconds, result.step_ms, result.scratch_mb = measure(
        candidate, model, xs, labels, args.repeats
    )
    _, after, _ = measure(baseline, reference_model, xs, labels, args.repeats)
    result.reference_ms = 0.5 * (before + after)
    result.speedup = result.reference_ms / result.step_ms
    return [result]


def run_one(name: str, args) -> Result:
    """Measure a single candidate in a fresh interpreter."""
    import subprocess
    import sys

    command = [
        sys.executable,
        __file__,
        f"--only={name}",
        f"--hidden={args.hidden}",
        f"--batch={args.batch}",
        f"--timesteps={args.timesteps}",
        f"--channels={args.channels}",
        f"--repeats={args.repeats}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            return Result(**json.loads(line))
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return Result(name, "", ok=False, error=detail[-1] if detail else "no output")


def _summarize(result: Result) -> str:
    if result.step_ms is None:
        return f"FAILED  {result.error}"
    mark = "exact" if result.exact else ("approx" if result.exact is False else "n/a")
    status = "" if result.ok else f"  REJECTED ({result.error})"
    return (
        f"{result.step_ms:7.2f} ms  {result.speedup:5.2f}x  {result.scratch_mb:7.1f} MB  "
        f"compile {result.compile_seconds:5.2f}s  {mark}{status}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--only", default="", help="measure one candidate and emit it as JSON")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.only:
        result = search(args)[0]
        print(json.dumps(dataclasses.asdict(result)))
        return

    print(f"device: {jax.devices()[0]}")
    print(f"batch {args.batch}  T {args.timesteps}  hidden {args.hidden}")
    results = []
    for candidate in candidates(args.timesteps):
        result = run_one(candidate.name, args)
        results.append(result)
        print(f"  {result.name:<20} {_summarize(result)}")

    accepted = [r for r in results if r.ok and r.speedup is not None]
    accepted.sort(key=lambda r: -r.speedup)

    # The reference measured against itself bounds what a ratio can mean.
    self_ratio = next((r.speedup for r in results if r.name == REFERENCE), 1.0)
    floor = abs(self_ratio - 1.0)
    print(f"\nnoise floor (reference against itself): {floor:.1%}")

    print(f"\n{'candidate':<20} {'step':>9} {'vs ref':>8} {'scratch':>10} {'match':>8}  verdict")
    for result in accepted:
        mark = "exact" if result.exact else ("approx" if result.exact is False else "n/a")
        if abs(result.speedup - 1.0) <= floor:
            verdict = "tie"
        else:
            verdict = "faster" if result.speedup > 1.0 else "slower"
        print(
            f"{result.name:<20} {result.step_ms:8.2f}ms {result.speedup:7.2f}x "
            f"{result.scratch_mb:9.1f}MB {mark:>8}  {verdict}"
        )

    rejected = [r for r in results if not r.ok]
    if rejected:
        print("\nrejected:")
        for result in rejected:
            print(f"  {result.name:<20} {result.error}")

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
