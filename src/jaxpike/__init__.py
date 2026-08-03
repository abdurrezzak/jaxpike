"""jaxpike -- fast, flexible spiking neural networks in JAX."""

from __future__ import annotations

from .conv import Conv2d, Flatten, Pool2d
from .execution import density, spike_count, spike_rate, unroll, unroll_checkpointed
from .init import lif_gain
from .layers import Dense, Sequential
from .neurons import ALIF, LIF, ALIFState, LIFState, LinearLIF
from .parallel import unroll_parallel
from .surrogate import ATan, Boxcar, FastSigmoid, Surrogate, Triangular

__version__ = "0.0.1.dev0"

__all__ = [
    "ALIF",
    "LIF",
    "ALIFState",
    "ATan",
    "Boxcar",
    "Conv2d",
    "Dense",
    "FastSigmoid",
    "Flatten",
    "LIFState",
    "LinearLIF",
    "Pool2d",
    "Sequential",
    "Surrogate",
    "Triangular",
    "__version__",
    "density",
    "lif_gain",
    "spike_count",
    "spike_rate",
    "unroll",
    "unroll_checkpointed",
    "unroll_parallel",
]
