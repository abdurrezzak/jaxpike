"""Aggregate a multi-seed accuracy sweep into mean, spread and range.

python benchmarks/summarize_seeds.py results/accuracy-seeds.txt
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics

RUN = re.compile(r"^\$ benchmarks/(\S+)(.*)$")
ACCURACY = re.compile(r"TEST ACCURACY:\s*([0-9.]+)")


def label(script: str, flags: str) -> str:
    if "torch_shd" in script:
        library = re.search(r"--library=(\S+)", flags)
        backend = re.search(r"--backend=(\S+)", flags)
        name = library.group(1) if library else "torch"
        return f"{name}:{backend.group(1)}" if backend and name == "spikingjelly" else name
    variant = re.search(r"--variant=(\S+)", flags)
    return f"jaxpike {variant.group(1)}" if variant else "jaxpike"


def parse(path: pathlib.Path) -> dict[str, list[float]]:
    runs: dict[str, list[float]] = {}
    current = None
    for line in path.read_text().splitlines():
        match = RUN.match(line)
        if match:
            current = label(*match.groups())
            continue
        found = ACCURACY.search(line)
        if found and current:
            runs.setdefault(current, []).append(float(found.group(1)))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=pathlib.Path)
    args = ap.parse_args()

    runs = parse(args.path)
    width = max((len(name) for name in runs), default=10)
    print(f"{'model':<{width}}  {'seeds':>5}  {'mean':>7}  {'sd':>6}  {'min':>7}  {'max':>7}")
    for name, values in sorted(runs.items(), key=lambda kv: -statistics.fmean(kv[1])):
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"{name:<{width}}  {len(values):>5}  {statistics.fmean(values):>7.4f}  "
            f"{sd:>6.4f}  {min(values):>7.4f}  {max(values):>7.4f}"
        )


if __name__ == "__main__":
    main()
