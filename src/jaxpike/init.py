"""Initialization helpers for spiking networks.

Deep SNNs have a failure mode that ANNs do not: they go **silent**. If a layer's drive lands
below threshold it fires less than the one before it, so activity decays multiplicatively with
depth until nothing reaches the output, and a silent network has no gradient anywhere to
recover from.

The cause is specific and fixable. The normalized convention ``v[t] = a*v[t-1] + (1-a)*x[t]``
is an exponential moving average, which attenuates signal standard deviation by
``sqrt((1-a)/(1+a))`` -- a factor of 6.3 at tau=20. Weights initialized for unit-variance
*activations* therefore produce membrane potentials six times smaller than intended, sitting
far below threshold.

`lif_gain` returns the compensating factor, and `Dense`/`Conv2d` accept it as `gain`.
"""

from __future__ import annotations

import math


def lif_gain(tau: float, dt: float = 1.0) -> float:
    """Weight multiplier that restores unit membrane variance for a LIF with this `tau`.

    The membrane is an exponential moving average with ``a = exp(-dt/tau)``. For iid input of
    variance s^2 its output variance is ``s^2 * (1-a)/(1+a)``, so multiplying weights by
    ``sqrt((1+a)/(1-a))`` cancels the attenuation and leaves the membrane on the same scale as
    `threshold`.

    Larger `tau` averages over a longer window and therefore needs more gain: 6.33 at tau=20,
    20.0 at tau=200.

    >>> round(lif_gain(20.0), 2)
    6.33
    """
    if tau <= dt:
        raise ValueError(f"tau must exceed dt={dt}, got {tau}")
    a = math.exp(-dt / tau)
    return math.sqrt((1.0 + a) / (1.0 - a))


__all__ = ["lif_gain"]
