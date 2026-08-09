"""jaxpike against snnTorch, SpikingJelly and Norse -- one container, one GPU.

Every library trains the same network on the same SHD arrays with the same optimizer, so the
measurement is of implementations rather than of hardware or of models. Each run is a separate
subprocess: JAX and PyTorch both want the whole device, and neither should be measured while
the other holds memory.

    modal run benchmarks/gpu/run_torch_comparison.py --suite speed --epochs 20 --trials 3
    modal run benchmarks/gpu/run_torch_comparison.py --suite accuracy --epochs 100
    modal run benchmarks/gpu/run_torch_comparison.py --suite scaling --epochs 10

Run one suite at a time. Every `modal run` of this file creates an ephemeral app under the
same name, and starting a second one stops the first mid-flight -- results already committed
to the volume survive, everything in progress does not.
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).parent.parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "cupy-cuda12x",
        "snntorch>=0.9.1",
        "spikingjelly==0.0.0.0.14",
        "norse>=1.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "jax[cuda12]>=0.4.35",
        "equinox>=0.11.4",
        "optax>=0.2.2",
        "jaxtyping>=0.2.28",
        "h5py>=3.11",
        "numpy>=1.26,<2.2",
    )
    .add_local_dir(REPO / "src", "/root/src")
    .add_local_dir(REPO / "benchmarks", "/root/benchmarks")
)

app = modal.App("jaxpike-vs-torch", image=image)

GPU = "T4"
CACHE = modal.Volume.from_name("shd-data", create_if_missing=True)

# JAX preallocates 75% of the device by default, which would survive into nothing here (each
# run is its own process) but makes the peak-memory numbers meaningless.
JAX_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def _run(cmd: list[str], tag: str, env: dict[str, str] | None = None) -> str:
    """Run one benchmark in its own process, appending output to the volume as it lands."""
    import pathlib
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, *cmd],
        cwd="/root",
        env={"PYTHONPATH": "/root:/root/src", "HOME": "/root", **(env or {})},
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


def _jaxpike(variant: str, tag: str, extra: list[str]) -> str:
    return _run(["benchmarks/spyx_shd.py", f"--variant={variant}", *extra], tag, JAX_ENV)


ALL_TORCH = "spikingjelly:cupy,spikingjelly:torch,snntorch:torch,norse:torch"


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def speed(
    epochs: int = 20,
    trials: int = 3,
    hidden: int = 128,
    batches: str = "64,128,256",
    libraries: str = ALL_TORCH,
    variants: str = "sequential,checkpointed,parallel",
) -> str:
    """`libraries` selects the PyTorch rows as `name:backend`; the slow ones cost minutes."""
    out = []
    for batch in [int(b) for b in batches.split(",")]:
        common = [f"--hidden={hidden}", f"--batch={batch}", f"--epochs={epochs}"]
        common += [f"--trials={trials}", "--skip-accuracy"]
        for spec in filter(None, libraries.split(",")):
            library, _, backend = spec.partition(":")
            out.append(
                _run(
                    [
                        "benchmarks/torch_shd.py",
                        f"--library={library}",
                        f"--backend={backend or 'torch'}",
                        *common,
                    ],
                    "speed",
                )
            )
        for variant in filter(None, variants.split(",")):
            out.append(_jaxpike(variant, "speed", common))
    return "\n".join(out)


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def accuracy(epochs: int = 100, hidden: int = 128, batch: int = 256) -> str:
    out = []
    common = [f"--hidden={hidden}", f"--batch={batch}", f"--epochs={epochs}"]
    for library, backend in (("spikingjelly", "cupy"), ("snntorch", "cupy"), ("norse", "cupy")):
        out.append(
            _run(
                [
                    "benchmarks/torch_shd.py",
                    f"--library={library}",
                    f"--backend={backend}",
                    *common,
                ],
                "accuracy-torch",
            )
        )
    for variant in ("sequential", "parallel"):
        out.append(_jaxpike(variant, "accuracy-torch", common))
    return "\n".join(out)


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def scaling(
    epochs: int = 10,
    trials: int = 3,
    timesteps_list: str = "256,512,1024",
    libraries: str = "spikingjelly:cupy",
    variants: str = "sequential,checkpointed,parallel",
) -> str:
    out = []
    for timesteps in [int(t) for t in timesteps_list.split(",")]:
        common = [f"--timesteps={timesteps}", f"--epochs={epochs}", f"--trials={trials}"]
        common += ["--skip-accuracy"]
        for spec in filter(None, libraries.split(",")):
            library, _, backend = spec.partition(":")
            out.append(
                _run(
                    [
                        "benchmarks/torch_shd.py",
                        f"--library={library}",
                        f"--backend={backend or 'torch'}",
                        *common,
                    ],
                    "scaling-torch",
                )
            )
        for variant in filter(None, variants.split(",")):
            out.append(_jaxpike(variant, "scaling-torch", common))
    return "\n".join(out)


@app.function(gpu=GPU, timeout=6 * 60 * 60, volumes={"/root/data": CACHE})
def jaxpike(
    epochs: int = 20,
    trials: int = 3,
    hidden: int = 128,
    batches: str = "64,128,256",
    variants: str = "sequential,checkpointed,parallel",
) -> str:
    """Re-measure only our side. The PyTorch rows cost minutes each and do not move."""
    out = []
    for batch in [int(b) for b in batches.split(",")]:
        common = [f"--hidden={hidden}", f"--batch={batch}", f"--epochs={epochs}"]
        common += [f"--trials={trials}", "--skip-accuracy"]
        for variant in variants.split(","):
            out.append(_jaxpike(variant, "jaxpike", common))
    return "\n".join(out)


@app.function(gpu=GPU, timeout=4 * 60 * 60, volumes={"/root/data": CACHE})
def autosearch(batch: int = 256, timesteps: int = 256, hidden: int = 128) -> str:
    """Verify-then-measure every execution strategy, and rank them."""
    return _run(
        [
            "benchmarks/autosearch.py",
            f"--batch={batch}",
            f"--timesteps={timesteps}",
            f"--hidden={hidden}",
            "--out=/root/data/results/autosearch.json",
        ],
        "autosearch",
        JAX_ENV,
    )


@app.function(gpu=GPU, timeout=2 * 60 * 60, volumes={"/root/data": CACHE})
def scan_unroll(factors: str = "1,2,4,8,16,32,64", variant: str = "sequential") -> str:
    """Calibrate how many timesteps the neuron loop should emit per iteration."""
    return _run(
        ["benchmarks/scan_unroll.py", f"--factors={factors}", f"--variant={variant}"],
        "scan-unroll",
        JAX_ENV,
    )


@app.function(gpu=GPU, timeout=60 * 60, volumes={"/root/data": CACHE})
def smoke() -> str:
    """Cheap end-to-end check that every library imports, runs and trains on this image."""
    out = []
    flags = ["--smoke", "--epochs=2", "--timesteps=64", "--batch=64", "--trials=1"]
    for library, backend in (
        ("spikingjelly", "cupy"),
        ("spikingjelly", "torch"),
        ("snntorch", "cupy"),
        ("norse", "cupy"),
    ):
        out.append(
            _run(
                ["benchmarks/torch_shd.py", f"--library={library}", f"--backend={backend}", *flags],
                "smoke",
            )
        )
    out.append(_jaxpike("sequential", "smoke", flags))
    return "\n".join(out)


@app.local_entrypoint()
def main(
    suite: str = "smoke",
    epochs: int = 20,
    trials: int = 3,
    batches: str = "64,128,256",
    timesteps: str = "256,512,1024",
    libraries: str = ALL_TORCH,
) -> None:
    suites = {
        "smoke": lambda: smoke.remote(),
        "scan-unroll": lambda: scan_unroll.remote(),
        "autosearch": lambda: autosearch.remote(timesteps=int(timesteps.split(",")[0])),
        "jaxpike": lambda: jaxpike.remote(epochs=epochs, trials=trials, batches=batches),
        "speed": lambda: speed.remote(
            epochs=epochs, trials=trials, batches=batches, libraries=libraries
        ),
        "accuracy": lambda: accuracy.remote(epochs=epochs),
        "scaling": lambda: scaling.remote(
            epochs=epochs, trials=trials, timesteps_list=timesteps, libraries=libraries
        ),
    }
    print(suites[suite]())
