"""jaxpike -- fast, flexible spiking neural networks in JAX."""

from __future__ import annotations

from .execution import density, spike_count, spike_rate, unroll
from .layers import Dense, Sequential
from .neurons import ALIF, LIF, ALIFState, LIFState
from .surrogate import ATan, Boxcar, FastSigmoid, Surrogate, Triangular

__version__ = "0.0.1.dev0"

__all__ = [
    "ALIF",
    "LIF",
    "ALIFState",
    "ATan",
    "Boxcar",
    "Dense",
    "FastSigmoid",
    "LIFState",
    "Sequential",
    "Surrogate",
    "Triangular",
    "__version__",
    "density",
    "spike_count",
    "spike_rate",
    "unroll",
]
