"""Automated optimization search over execution strategies.

Every candidate is a way of running the same network. The loop is deliberately dumb and
strict, because the failure mode of hand-tuned performance work is believing a speedup that
either did not happen or was not free:

    1. Correctness gate first. A candidate is compared against the reference implementation on
       outputs *and* gradients, and is rejected outright if it disagrees by more than its
       declared tolerance. Whether it was *bit*-identical is reported separately, because
       reassociation inside a fused loop body perturbs results at around 1e-9 -- harmless, but
       not nothing, and worth never claiming otherwise.
    2. Then measurement. Compile and steady state are timed separately, because they are paid
       on different schedules and only one of them is what a training run spends its time on.
    3. Then ranking, including the losers. A search that records only its wins cannot tell you
       what has already been ruled out.

Results are written as JSON so a later run can compare against an earlier one rather than
against memory.

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
    scratch_mb: float | None = None
    error: str | None = None


def candidates(timesteps: int) -> list[Candidate]:
    """The search space. Extend here; the loop needs no other change."""
    space = [
        Candidate(REFERENCE, jp.unroll, note="reference"),
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
    for chunk in sorted({c for c in (8, 16, 32, 64) if timesteps % c == 0}):
        space.append(
            Candidate(
                f"checkpointed-{chunk}",
                lambda m, xs, c=chunk: jp.unroll_checkpointed(m, xs, chunk_size=c),
                note=f"rematerialized in chunks of {chunk}",
            )
        )
    return space


def make_step(model, runner, optimizer):
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(p, xs, labels):
        traces, _ = runner(eqx.combine(p, static), xs)
        return integral_crossentropy(traces, labels)

    def step(p, opt_state, xs, labels):
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


def measure(candidate: Candidate, model, xs, labels, repeats: int) -> tuple[float, float, float]:
    optimizer = optax.adam(5e-4)
    params, _, _, step = make_step(model, candidate.runner, optimizer)
    opt_state = optimizer.init(params)
    compiled = eqx.filter_jit(step)

    start = time.perf_counter()
    out = compiled(params, opt_state, xs, labels)
    jax.block_until_ready(out)
    compile_seconds = time.perf_counter() - start

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = compiled(params, opt_state, xs, labels)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)

    scratch = compile_stats(step, params, opt_state, xs, labels)["temp_bytes"]
    return compile_seconds, float(np.median(times)) * 1e3, scratch / 2**20


def search(args) -> list[Result]:
    key = jax.random.key(0)
    xs = (jax.random.uniform(key, (args.timesteps, args.batch, args.channels)) < 0.05).astype(
        jnp.float32
    )
    labels = jax.random.randint(key, (args.batch,), 0, 20)

    def model_for(neuron: str):
        return build(args.hidden, args.channels, key, neuron_kind=neuron)

    reference = reference_gradients(model_for("lif"), xs, labels)

    results = []
    for candidate in candidates(args.timesteps):
        model = model_for(candidate.neuron)
        result = Result(candidate.name, candidate.note, ok=True)
        try:
            if candidate.tolerance != float("inf"):
                output_error, gradient_error = verify(candidate, model, xs, labels, reference)
                result.output_error = output_error
                result.gradient_error = gradient_error
                worst = max(output_error, gradient_error)
                result.exact = worst == 0.0
                if worst > candidate.tolerance:
                    result.ok = False
                    result.error = f"disagrees with reference by {worst:.3e}"
            result.compile_seconds, result.step_ms, result.scratch_mb = measure(
                candidate, model, xs, labels, args.repeats
            )
        except Exception as exc:  # a candidate that cannot run is a result, not a crash
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
        print(f"  {result.name:<20} {_summarize(result)}")
    return results


def _summarize(result: Result) -> str:
    if result.step_ms is None:
        return f"FAILED  {result.error}"
    mark = "exact" if result.exact else ("approx" if result.exact is False else "n/a")
    status = "" if result.ok else f"  REJECTED ({result.error})"
    return (
        f"{result.step_ms:7.2f} ms  {result.scratch_mb:7.1f} MB  "
        f"compile {result.compile_seconds:5.2f}s  {mark}{status}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=256)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"device: {jax.devices()[0]}")
    print(f"batch {args.batch}  T {args.timesteps}  hidden {args.hidden}")
    results = search(args)

    accepted = [r for r in results if r.ok and r.step_ms is not None]
    accepted.sort(key=lambda r: r.step_ms)
    baseline = next((r for r in results if r.name == REFERENCE), None)

    print(f"\n{'candidate':<20} {'step':>9} {'vs ref':>8} {'scratch':>10} {'match':>8}")
    for result in accepted:
        speedup = baseline.step_ms / result.step_ms if baseline and baseline.step_ms else 0.0
        mark = "exact" if result.exact else ("approx" if result.exact is False else "n/a")
        print(
            f"{result.name:<20} {result.step_ms:8.2f}ms {speedup:7.2f}x "
            f"{result.scratch_mb:9.1f}MB {mark:>8}"
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
