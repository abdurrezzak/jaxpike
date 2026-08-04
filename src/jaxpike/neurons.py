"""Reference neuron models.

Every neuron follows the same contract, which is what lets the execution engine treat them
uniformly:

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

# float32 even under bf16/fp16 training: a leaky integrator runs for thousands of steps, and
# low-precision accumulation drifts enough to change which neurons cross threshold.
STATE_DTYPE = jnp.float32


class LIFState(eqx.Module):
    v: Float[Array, "..."]


class LIF(eqx.Module):
    """Leaky integrate-and-fire.

        v[t] = alpha*v[t-1] + (1 - alpha)*x[t],   alpha = exp(-dt/tau)
        s[t] = H(v[t] - threshold)

    followed by a reset: ``subtract`` removes one threshold from the membrane (retaining the
    overshoot, so information is not discarded) while ``zero`` clamps it back to rest.

    The ``(1 - alpha)`` on the input is the normalized convention: a constant drive `x` settles
    at exactly `x`, so inputs are in the same units as `threshold`. A single timestep therefore
    injects only ``1 - alpha`` of the input, about 5% at tau=20. snnTorch omits this factor, so
    weights ported from it need rescaling by ``1/(1 - alpha)``.

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
    needing two state variables, so it is the worked example for a multi-variable neuron.
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

    Dropping the reset costs self-regulation -- a strongly driven unit saturates at one spike
    per timestep -- and buys an affine recurrence, so the whole time axis can be solved with an
    associative scan instead of a sequential loop. See `jaxpike.unroll_parallel`.

    This is the PSN-style neuron of the parallel spiking network literature.
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


class LeakyIntegrator(eqx.Module):
    """A LIF that never spikes: outputs membrane potential directly.

        v[t] = alpha*v[t-1] + (1 - alpha)*x[t]

    The standard SNN readout layer. Classifying on spike counts means the loss only sees a
    unit once it crosses threshold, so a class that never fires produces no gradient and can
    never learn to fire; reading the continuous membrane keeps every output unit
    differentiable from the first step.

    Being linear and reset-free, it also parallelizes over time.
    """

    log_tau: Float[Array, "..."]
    dt: float = eqx.field(static=True)

    def __init__(self, tau: float | Array = 20.0, *, dt: float = 1.0):
        tau_arr = jnp.asarray(tau, dtype=STATE_DTYPE)
        if jnp.any(tau_arr <= dt):
            raise ValueError(
                f"tau must exceed dt={dt} or the discretization is unstable; got tau={tau}"
            )
        self.log_tau = jnp.log(tau_arr)
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
        return LIFState(v=v), v

    def parallel_apply(self, state: LIFState, xs: Array) -> tuple[LIFState, Array]:
        from .parallel import scan_linear_recurrence

        alpha = self.alpha
        xs = xs.astype(STATE_DTYPE)
        a = jnp.broadcast_to(alpha, xs.shape)
        v = scan_linear_recurrence(a, (1.0 - alpha) * xs, state.v)
        return LIFState(v=v[-1]), v


# Firing-pattern presets from Izhikevich (2003), "Simple Model of Spiking Neurons".
IZHIKEVICH_PRESETS: dict[str, tuple[float, float, float, float]] = {
    #                      a      b      c     d
    "regular_spiking": (0.02, 0.20, -65.0, 8.0),
    "intrinsically_bursting": (0.02, 0.20, -55.0, 4.0),
    "chattering": (0.02, 0.20, -50.0, 2.0),
    "fast_spiking": (0.10, 0.20, -65.0, 2.0),
    "low_threshold_spiking": (0.02, 0.25, -65.0, 2.0),
    "resonator": (0.10, 0.26, -65.0, 2.0),
    "thalamo_cortical": (0.02, 0.25, -65.0, 0.05),
}


class IzhikevichState(eqx.Module):
    v: Float[Array, "..."]
    u: Float[Array, "..."]


class Izhikevich(eqx.Module):
    """Izhikevich neuron -- two variables, a quadratic term, and most of cortex's firing zoo.

        v' = 0.04*v^2 + 5*v + 140 - u + I
        u' = a*(b*v - u)
        if v >= 30 mV:  v <- c,  u <- u + d

    Unlike LIF this is a *spike-generating* model rather than a threshold-crossing one: the
    quadratic term produces a genuine upstroke, and 30 mV detects the spike at its peak rather
    than acting as a threshold in the LIF sense. Construct by name with
    `Izhikevich.preset("chattering")`.

    Three practical notes, because this model is less forgiving than LIF:

    Voltages are in millivolts, not the dimensionless units the LIF classes use. Resting
    potential is around -65 mV and input currents are on the order of 1-20, so weight
    initialization tuned for LIF does nothing useful here.

    The quadratic term makes forward Euler unstable at dt=1: v can run away to infinity within
    a step for large input. Sub-stepping (two half steps, as Izhikevich's own code does) plus a
    clamp keeps it finite without changing the dynamics below the spike.

    It is nonlinear in time, so there is no `parallel_apply` and `unroll_parallel` will raise.
    """

    log_tau_unused: Float[Array, "..."]  # placeholder keeps the pytree non-empty for grads
    surrogate: Surrogate
    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    c: float = eqx.field(static=True)
    d: float = eqx.field(static=True)
    v_peak: float = eqx.field(static=True)
    v_scale: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    substeps: int = eqx.field(static=True)

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        *,
        v_peak: float = 30.0,
        v_scale: float = 5.0,
        dt: float = 1.0,
        substeps: int = 2,
        surrogate: Surrogate | None = None,
    ):
        if substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {substeps}")
        self.log_tau_unused = jnp.zeros(())
        self.surrogate = surrogate if surrogate is not None else FastSigmoid()
        self.a, self.b, self.c, self.d = a, b, c, d
        self.v_peak, self.v_scale = v_peak, v_scale
        self.dt, self.substeps = dt, substeps

    @classmethod
    def preset(cls, name: str, **kwargs) -> Izhikevich:
        """Construct by firing pattern, e.g. ``Izhikevich.preset("fast_spiking")``."""
        if name not in IZHIKEVICH_PRESETS:
            raise ValueError(f"unknown preset {name!r}; available: {sorted(IZHIKEVICH_PRESETS)}")
        a, b, c, d = IZHIKEVICH_PRESETS[name]
        return cls(a, b, c, d, **kwargs)

    def init_state(self, input_shape: tuple[int, ...]) -> IzhikevichState:
        v = jnp.full(input_shape, self.c, dtype=STATE_DTYPE)
        return IzhikevichState(v=v, u=self.b * v)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state: IzhikevichState, x: Array) -> tuple[IzhikevichState, Array]:
        v, u = state.v, state.u
        x = x.astype(STATE_DTYPE)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            dv = 0.04 * v * v + 5.0 * v + 140.0 - u + x
            v = v + h * dv
            # Clamp above the detection level, not below it: the dynamics up to the spike are
            # untouched, but a runaway quadratic cannot reach inf and poison the gradient.
            v = jnp.clip(v, -200.0, self.v_peak + 10.0)
        u = u + self.dt * self.a * (self.b * v - u)

        # v_scale converts the millivolt gap into the dimensionless input surrogates expect;
        # without it a slope-25 surrogate sees tens of millivolts and returns no gradient.
        s = self.surrogate((v - self.v_peak) / self.v_scale)
        v = v + (self.c - v) * s  # v <- c where it spiked
        u = u + self.d * s
        return IzhikevichState(v=v, u=u), s


__all__ = [
    "ALIF",
    "IZHIKEVICH_PRESETS",
    "LIF",
    "STATE_DTYPE",
    "ALIFState",
    "Izhikevich",
    "IzhikevichState",
    "LIFState",
    "LeakyIntegrator",
    "LinearLIF",
]
