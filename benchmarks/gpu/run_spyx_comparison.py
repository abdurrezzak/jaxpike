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


def _run(cmd: list[str], tag: str = "run") -> str:
    """Run one benchmark and append its output to the shared volume immediately.

    `modal run` blocks on the result, so a dropped client connection kills the call and takes
    every buffered result with it. Writing each result to the volume as it lands makes a run
    recoverable.
    """
    import pathlib
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
    text = f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr}"

    results = pathlib.Path("/root/data/results")
    results.mkdir(parents=True, exist_ok=True)
    with (results / f"{tag}.txt").open("a") as handle:
        handle.write(text + "\n")
    CACHE.commit()
    return text


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
                ],
                "accuracy",
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
                    "--skip-accuracy",
                ],
                "compare",
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
                        "--skip-accuracy",
                    ],
                    "compare",
                )
            )
    return "\n".join(out)


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def scaling(
    epochs: int = 10,
    trials: int = 3,
    timesteps_list: str = "256,512,1024",
    include_spyx: bool = True,
) -> str:
    """Where jaxpike is expected to pull away: their scan is linear in T, ours is not."""
    out = []
    for timesteps in [int(t) for t in timesteps_list.split(",")]:
        if include_spyx:
            out.append(
                _run(
                    [
                        "benchmarks/spyx_reference.py",
                        f"--timesteps={timesteps}",
                        f"--epochs={epochs}",
                        f"--trials={trials}",
                        "--skip-accuracy",
                    ],
                    "scaling",
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
                        "--skip-accuracy",
                    ],
                    "scaling",
                )
            )
    return "\n".join(out)


@app.local_entrypoint()
def main(
    suite: str = "accuracy",
    epochs: int = 100,
    trials: int = 5,
    variants: str = "sequential,parallel",
    batches: str = "64,128,256",
    timesteps: str = "256,512,1024",
    include_spyx: bool = True,
    spawn: bool = False,
) -> None:
    """`--spawn` queues the run server-side and returns immediately.

    `.remote()` blocks until the function returns, so the run dies with the client -- a
    dropped connection or a shell that exits takes hours of GPU time with it. Results land on
    the volume either way; with `--spawn` nothing local needs to stay alive to collect them.
    """
    suites = {
        "accuracy": lambda: accuracy.remote(epochs=epochs, variants=variants),
        "compare": lambda: compare.remote(epochs=epochs, trials=trials, batches=batches),
        "scaling": lambda: scaling.remote(
            epochs=epochs,
            trials=trials,
            timesteps_list=timesteps,
            include_spyx=include_spyx,
        ),
    }
    if spawn:
        fns = {"accuracy": accuracy, "compare": compare, "scaling": scaling}
        kwargs = {
            "accuracy": dict(epochs=epochs, variants=variants),
            "compare": dict(epochs=epochs, trials=trials, batches=batches),
            "scaling": dict(
                epochs=epochs,
                trials=trials,
                timesteps_list=timesteps,
                include_spyx=include_spyx,
            ),
        }[suite]
        call = fns[suite].spawn(**kwargs)
        print(f"queued {suite} as {call.object_id}; results land in the shd-data volume")
        return
    print(suites[suite]())
