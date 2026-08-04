"""jaxpike against Spyx on the SHD benchmark, both in one container on one GPU.

The Spyx paper's Table 1 was measured on an RTX A6000. Quoting those numbers next to anything
measured elsewhere would compare hardware, not libraries, so this runs both here:

    modal run benchmarks/gpu/run_spyx_comparison.py --suite accuracy
    modal run benchmarks/gpu/run_spyx_comparison.py --suite compare --trials 2
    modal run benchmarks/gpu/run_spyx_comparison.py --suite scaling --epochs 10

Spyx is pinned to 0.1.19, the release contemporary with the paper; later versions moved to
Flax NNX and the benchmark notebook's API no longer exists.
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).parent.parent.parent

# jaxpike and paper-era Spyx share one JAX build so neither gets a different XLA. The JAX
# spec matches run_modal.py; pinning it to 0.4.35 exactly pulls an nvidia-cuda-nvcc wheel
# whose namespace package JAX then fails to resolve (`cuda_nvcc.__file__` is None).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]>=0.4.35",
        "equinox>=0.11.4",
        "optax>=0.2.2",
        "jaxtyping>=0.2.28",
        "h5py>=3.11",
        "numpy>=1.26",
        "spyx==0.1.19",
        "dm-haiku>=0.0.12",
        "jax-tqdm",
    )
    .add_local_dir(REPO / "src", "/root/src")
    .add_local_dir(REPO / "benchmarks", "/root/benchmarks")
)

app = modal.App("jaxpike-vs-spyx", image=image)

# Modal's free tier allows T4 without a payment method; A100/H100 do not.
GPU = "T4"
CACHE = modal.Volume.from_name("shd-data", create_if_missing=True)


def _run(cmd: list[str]) -> str:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, *cmd],
        cwd="/root",
        env={"PYTHONPATH": "/root:/root/src", "HOME": "/root"},
        capture_output=True,
        text=True,
        check=False,
    )
    return f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr}"


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def accuracy(
    epochs: int = 100,
    hidden: int = 128,
    batch: int = 256,
    variants: str = "sequential,parallel",
) -> str:
    """Does jaxpike reach Spyx's reported 70-75% band under their exact protocol?"""
    out = []
    for variant in variants.split(","):
        out.append(
            _run(
                [
                    "benchmarks/spyx_shd.py",
                    f"--variant={variant}",
                    f"--epochs={epochs}",
                    f"--hidden={hidden}",
                    f"--batch={batch}",
                ]
            )
        )
    return "\n".join(out)


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def compare(
    epochs: int = 100, trials: int = 2, hidden: int = 128, batches: str = "64,128,256"
) -> str:
    """Both libraries, every batch size, in one container so the GPU is literally the same."""
    out = []
    for batch in [int(b) for b in batches.split(",")]:
        out.append(
            _run(
                [
                    "benchmarks/spyx_reference.py",
                    f"--hidden={hidden}",
                    f"--batch={batch}",
                    f"--epochs={epochs}",
                    f"--trials={trials}",
                ]
            )
        )
        for variant in ("sequential", "checkpointed", "parallel"):
            out.append(
                _run(
                    [
                        "benchmarks/spyx_shd.py",
                        f"--variant={variant}",
                        f"--hidden={hidden}",
                        f"--batch={batch}",
                        f"--epochs={epochs}",
                        f"--trials={trials}",
                    ]
                )
            )
    return "\n".join(out)


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def scaling(epochs: int = 10, trials: int = 3) -> str:
    """Where jaxpike is expected to pull away: their scan is linear in T, ours is not."""
    out = []
    for timesteps in (256, 512, 1024):
        out.append(
            _run(
                [
                    "benchmarks/spyx_reference.py",
                    f"--timesteps={timesteps}",
                    f"--epochs={epochs}",
                    f"--trials={trials}",
                ]
            )
        )
        for variant in ("sequential", "parallel"):
            out.append(
                _run(
                    [
                        "benchmarks/spyx_shd.py",
                        f"--variant={variant}",
                        f"--timesteps={timesteps}",
                        f"--epochs={epochs}",
                        f"--trials={trials}",
                    ]
                )
            )
    return "\n".join(out)


@app.local_entrypoint()
def main(
    suite: str = "accuracy",
    epochs: int = 100,
    trials: int = 5,
    variants: str = "sequential,parallel",
) -> None:
    suites = {
        "accuracy": lambda: accuracy.remote(epochs=epochs, variants=variants),
        "compare": lambda: compare.remote(epochs=epochs, trials=trials),
        "scaling": lambda: scaling.remote(epochs=epochs, trials=trials),
    }
    print(suites[suite]())
