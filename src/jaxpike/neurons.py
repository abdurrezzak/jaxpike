"""Reference neuron models.

These are the correctness reference for the whole library: pure JAX, no kernels, no tricks.
Phase 2's fused Pallas implementations are validated against these forever, so clarity here
matters more than speed.

Every neuron follows the same contract, which is what lets the execution engine and the
codegen path treat them uniformly:

    init_state(input_shape) -> state pytree
    out_shape(input_shape)  -> output shape
    __call__(state, x)      -> (new_state, spikes)

State is explicit and functional. Nothing is stored on the module between steps.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .surrogate import FastSigmoid, Surrogate

# Membrane state is accumulated in float32 even under bf16/fp16 training: a leaky integrator
# runs for thousands of steps and low-precision accumulation drifts enough to change which
# neurons cross threshold.
STATE_DTYPE = jnp.float32


class LIFState(eqx.Module):
    v: Float[Array, "..."]


class LIF(eqx.Module):
    """Leaky integrate-and-fire.

        v[t] = alpha*v[t-1] + (1 - alpha)*x[t],   alpha = exp(-dt/tau)
        s[t] = H(v[t] - threshold)

    followed by a reset: ``subtract`` removes one threshold from the membrane (retaining the
    overshoot, so information is not discarded) while ``zero`` clamps it back to rest.

    Note the ``(1 - alpha)`` on the input, which is the normalized convention: a constant
    drive `x` drives the membrane to a steady state of exactly `x`, so inputs are expressed in
    the same units as `threshold` and a drive below threshold provably never fires. The
    practical consequence is that a *single* timestep injects only ``1 - alpha`` of the input
    (about 5% at tau=20), so transient drives must be large to fire on their own. snnTorch
    omits this factor, so weights ported from it need rescaling by ``1/(1 - alpha)``.

    `tau` is stored as its log so it stays positive under unconstrained optimization, and it
    is a learnable leaf by default. Freeze it with `equinox.partition` if you don't want that.
    """

    log_tau: Float[Array, "..."]
    surrogate: Surrogate
    threshold: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    reset: str = eqx.field(static=True)

    def __init__(
        self,
        tau: float | Array = 20.0,
        *,
        threshold: float = 1.0,
        dt: float = 1.0,
        reset: str = "subtract",
        surrogate: Surrogate | None = None,
    ):
        if reset not in ("subtract", "zero"):
            raise ValueError(f"reset must be 'subtract' or 'zero', got {reset!r}")
        tau_arr = jnp.asarray(tau, dtype=STATE_DTYPE)
        if jnp.any(tau_arr <= dt):
            raise ValueError(
                f"tau must exceed dt={dt} or the discretization is unstable; got tau={tau}"
            )
        self.log_tau = jnp.log(tau_arr)
        self.surrogate = surrogate if surrogate is not None else FastSigmoid()
        self.threshold = threshold
        self.dt = dt
        self.reset = reset

    @property
    def tau(self) -> Array:
        return jnp.exp(self.log_tau)

    @property
    def alpha(self) -> Array:
        return jnp.exp(-self.dt / self.tau)

    def init_state(self, input_shape: tuple[int, ...]) -> LIFState:
        return LIFState(v=jnp.zeros(input_shape, dtype=STATE_DTYPE))

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state: LIFState, x: Array) -> tuple[LIFState, Array]:
        alpha = self.alpha
        v = alpha * state.v + (1.0 - alpha) * x.astype(STATE_DTYPE)
        s = self.surrogate(v - self.threshold)
        v = v - self.threshold * s if self.reset == "subtract" else v * (1.0 - s)
        return LIFState(v=v), s


class ALIFState(eqx.Module):
    v: Float[Array, "..."]
    a: Float[Array, "..."]


class ALIF(eqx.Module):
    """Adaptive LIF: each spike raises the neuron's own threshold, which then decays.

    The effective threshold is ``threshold + beta*a``, where the adaptation variable `a`
    integrates the neuron's own spikes with time constant `tau_a`. This is the smallest model
    that needs two state variables, so it is the one that proves the state contract
    generalizes past a single membrane array.
    """

    log_tau: Float[Array, "..."]
    log_tau_a: Float[Array, "..."]
    surrogate: Surrogate
    beta: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    reset: str = eqx.field(static=True)

    def __init__(
        self,
        tau: float | Array = 20.0,
        tau_a: float | Array = 200.0,
        *,
        beta: float = 1.8,
        threshold: float = 1.0,
        dt: float = 1.0,
        reset: str = "subtract",
        surrogate: Surrogate | None = None,
    ):
        if reset not in ("subtract", "zero"):
            raise ValueError(f"reset must be 'subtract' or 'zero', got {reset!r}")
        self.log_tau = jnp.log(jnp.asarray(tau, dtype=STATE_DTYPE))
        self.log_tau_a = jnp.log(jnp.asarray(tau_a, dtype=STATE_DTYPE))
        self.surrogate = surrogate if surrogate is not None else FastSigmoid()
        self.beta = beta
        self.threshold = threshold
        self.dt = dt
        self.reset = reset

    def init_state(self, input_shape: tuple[int, ...]) -> ALIFState:
        z = jnp.zeros(input_shape, dtype=STATE_DTYPE)
        return ALIFState(v=z, a=z)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state: ALIFState, x: Array) -> tuple[ALIFState, Array]:
        alpha = jnp.exp(-self.dt / jnp.exp(self.log_tau))
        rho = jnp.exp(-self.dt / jnp.exp(self.log_tau_a))
        v = alpha * state.v + (1.0 - alpha) * x.astype(STATE_DTYPE)
        thr = self.threshold + self.beta * state.a
        s = self.surrogate(v - thr)
        v = v - thr * s if self.reset == "subtract" else v * (1.0 - s)
        a = rho * state.a + s
        return ALIFState(v=v, a=a), s


class LinearLIF(eqx.Module):
    """Reset-free LIF. Identical to `LIF` except that a spike does not perturb the membrane.

        v[t] = alpha*v[t-1] + (1 - alpha)*x[t]
        s[t] = H(v[t] - threshold)

    Dropping the reset is a real modelling choice with a real cost -- the neuron cannot
    regulate its own firing, so a strongly driven unit saturates at one spike per timestep --
    and a real payoff: the recurrence stays affine, so the whole time axis can be solved with
    an associative scan instead of a sequential loop. That measured 119x faster than
    sequential at T=8192 on a T4.

    This is the PSN-style neuron of the parallel spiking network literature, and it is the
    tier-1 case for `jaxpike.unroll_parallel`.
    """

    log_tau: Float[Array, "..."]
    surrogate: Surrogate
    threshold: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(
        self,
        tau: float | Array = 20.0,
        *,
        threshold: float = 1.0,
        dt: float = 1.0,
        surrogate: Surrogate | None = None,
    ):
        tau_arr = jnp.asarray(tau, dtype=STATE_DTYPE)
        if jnp.any(tau_arr <= dt):
            raise ValueError(
                f"tau must exceed dt={dt} or the discretization is unstable; got tau={tau}"
            )
        self.log_tau = jnp.log(tau_arr)
        self.surrogate = surrogate if surrogate is not None else FastSigmoid()
        self.threshold = threshold
        self.dt = dt

    @property
    def alpha(self) -> Array:
        return jnp.exp(-self.dt / jnp.exp(self.log_tau))

    def init_state(self, input_shape: tuple[int, ...]) -> LIFState:
        return LIFState(v=jnp.zeros(input_shape, dtype=STATE_DTYPE))

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state: LIFState, x: Array) -> tuple[LIFState, Array]:
        alpha = self.alpha
        v = alpha * state.v + (1.0 - alpha) * x.astype(STATE_DTYPE)
        return LIFState(v=v), self.surrogate(v - self.threshold)

    def parallel_apply(self, state: LIFState, xs: Array) -> tuple[LIFState, Array]:
        """Solve every timestep at once via an associative scan over the affine recurrence."""
        from .parallel import scan_linear_recurrence

        alpha = self.alpha
        xs = xs.astype(STATE_DTYPE)
        a = jnp.broadcast_to(alpha, xs.shape)
        b = (1.0 - alpha) * xs
        v = scan_linear_recurrence(a, b, state.v)
        return LIFState(v=v[-1]), self.surrogate(v - self.threshold)


__all__ = ["ALIF", "LIF", "STATE_DTYPE", "ALIFState", "LIFState", "LinearLIF"]
