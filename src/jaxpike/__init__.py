"""jaxpike -- fast, flexible spiking neural networks in JAX."""

from __future__ import annotations

from .conv import Conv2d, Flatten, Pool2d
from .execution import density, spike_count, spike_rate, unroll, unroll_checkpointed
from .init import lif_gain
from .layers import Dense, Sequential
from .neurons import (
    ALIF,
    IZHIKEVICH_PRESETS,
    LIF,
    ALIFState,
    Izhikevich,
    IzhikevichState,
    LeakyIntegrator,
    LIFState,
    LinearLIF,
)
from .parallel import unroll_parallel
from .plasticity import STDP, STDPState, stdp_window
from .surrogate import ATan, Boxcar, FastSigmoid, Surrogate, Triangular
from .training import (
    accuracy,
    count_logits,
    cross_entropy,
    iterate_batches,
    make_step,
    max_membrane_logits,
    rate_penalty,
)

__version__ = "0.0.1.dev0"

__all__ = [
    "ALIF",
    "IZHIKEVICH_PRESETS",
    "LIF",
    "STDP",
    "ALIFState",
    "ATan",
    "Boxcar",
    "Conv2d",
    "Dense",
    "FastSigmoid",
    "Flatten",
    "Izhikevich",
    "IzhikevichState",
    "LIFState",
    "LeakyIntegrator",
    "LinearLIF",
    "Pool2d",
    "STDPState",
    "Sequential",
    "Surrogate",
    "Triangular",
    "__version__",
    "accuracy",
    "count_logits",
    "cross_entropy",
    "density",
    "iterate_batches",
    "lif_gain",
    "make_step",
    "max_membrane_logits",
    "rate_penalty",
    "spike_count",
    "spike_rate",
    "stdp_window",
    "unroll",
    "unroll_checkpointed",
    "unroll_parallel",
]
