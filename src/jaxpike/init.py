"""Initialization helpers for spiking networks.

Deep SNNs have a failure mode that ANNs do not: they go **silent**. Each layer emits binary
spikes at some rate, and if a layer's drive lands below threshold it fires less than the one
before it, so activity decays multiplicatively with depth until nothing reaches the output.
A silent network has no gradient anywhere and never recovers — it is not slow training, it is
no training.

Measured in this library with plain LeCun initialization, a three-layer spiking convnet at
threshold 0.2 fires at 0.350, then 0.033, then 0.000. Dead by layer three.

The cause is specific and fixable. Our neurons use the normalized convention
``v[t] = a*v[t-1] + (1-a)*x[t]``, which is an exponential moving average. For white-noise
input it attenuates signal standard deviation by ``sqrt((1-a)/(1+a))`` — a factor of 6.3 at
tau=20 — because averaging over a window cancels most of the fluctuation. Weights initialized
for unit-variance *activations* therefore produce membrane potentials six times smaller than
intended, sitting far below threshold.

`lif_gain` returns exactly the compensating factor, and `Dense`/`Conv2d` accept it as `gain`.
"""

from __future__ import annotations

import math


def lif_gain(tau: float, dt: float = 1.0) -> float:
    """Weight multiplier that restores unit membrane variance for a LIF with this `tau`.

    The membrane is an exponential moving average with ``a = exp(-dt/tau)``. For iid input of
    variance s^2 its output variance is ``s^2 * (1-a)/(1+a)``, so multiplying weights by
    ``sqrt((1+a)/(1-a))`` cancels the attenuation and leaves the membrane on the same scale as
    `threshold`.

    Larger `tau` averages over a longer window and therefore needs more gain: 6.3 at tau=20,
    20.0 at tau=200.

    >>> round(lif_gain(20.0), 2)
    6.32
    """
    if tau <= dt:
        raise ValueError(f"tau must exceed dt={dt}, got {tau}")
    a = math.exp(-dt / tau)
    return math.sqrt((1.0 + a) / (1.0 - a))


__all__ = ["lif_gain"]
