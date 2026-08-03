"""Run jaxpike benchmarks on a rented GPU via Modal.

Local macOS cannot run Pallas (Metal has no Triton/Mosaic backend), so every kernel and
timing measurement happens here. Usage:

    modal run benchmarks/gpu/run_modal.py                      # memory scaling, A100
    modal run benchmarks/gpu/run_modal.py --gpu L4 --bench all
    modal run benchmarks/gpu/run_modal.py --gpu A100-40GB --bench parallel

Costs are per-second and the container exits when the run finishes.
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]>=0.4.35",
        "equinox>=0.11.4",
        "optax>=0.2.2",
        "jaxtyping>=0.2.28",
    )
    .add_local_dir(REPO / "src", "/root/src")
    .add_local_dir(REPO / "benchmarks", "/root/benchmarks")
)

app = modal.App("jaxpike-benchmarks", image=image)


# T4 is the default because Modal's free tier allows it without a payment method on file;
# A100/H100 require one. Override with --gpu once billing is set up.
@app.function(gpu="T4", timeout=60 * 60)
def run(bench: str = "memory", args: list[str] | None = None) -> str:
    import subprocess
    import sys

    env = {"PYTHONPATH": "/root:/root/src"}
    scripts = {
        "memory": ["benchmarks/memory_scaling.py", "--time"],
        "parallel": ["benchmarks/parallel_scan.py"],
    }
    targets = list(scripts.values()) if bench == "all" else [scripts[bench]]

    out = []
    for target in targets:
        cmd = [sys.executable, *target, *(args or [])]
        result = subprocess.run(
            cmd, cwd="/root", env=env, capture_output=True, text=True, check=False
        )
        out.append(result.stdout + result.stderr)
    return "\n".join(out)


@app.local_entrypoint()
def main(bench: str = "memory", gpu: str = "A100", lengths: str = "") -> None:
    extra = ["--lengths", *lengths.split(",")] if lengths else []
    fn = run.with_options(gpu=gpu)
    print(fn.remote(bench=bench, args=extra))
