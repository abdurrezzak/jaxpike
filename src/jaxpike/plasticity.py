"""Spike-timing-dependent plasticity.

Everything else in this library trains by backpropagation through time with surrogate
gradients: a global loss, gradients pushed backwards through the whole unrolled network.
STDP is a different animal entirely and it is worth being clear about the difference before
using it.

STDP is **local and unsupervised**. A synapse changes strength based only on the relative
timing of the spikes at its two ends: pre shortly before post strengthens it, the reverse
weakens it. No loss function, no gradients, no backward pass. That locality is why
neuromorphic chips can implement it in hardware.

The trade is real: STDP has no notion of a task. It extracts correlation structure from input
statistics, but on its own it will not train a classifier the way BPTT will. The usual recipes
are STDP-pretraining followed by a supervised readout, or a three-factor rule where a reward
signal gates the update.

The implementation is the standard pair-based rule with exponential eligibility traces:

    trace_pre[t]  = exp(-dt/tau_pre) * trace_pre[t-1]   + pre_spikes[t]
    trace_post[t] = exp(-dt/tau_post) * trace_post[t-1] + post_spikes[t]

    dw += a_plus  * outer(post_spikes[t], trace_pre[t-1])     # post after pre -> strengthen
    dw -= a_minus * outer(trace_post[t-1], pre_spikes[t])     # pre after post -> weaken

Traces from `t-1` are used deliberately: a pre and post spike in the same timestep are
simultaneous, not causally ordered, and should not potentiate.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class STDPState(eqx.Module):
    """Per-neuron eligibility traces -- decaying memories of recent spiking."""

    pre: Float[Array, "B pre"]
    post: Float[Array, "B post"]


class STDP(eqx.Module):
    """Pair-based STDP with exponential traces.

    Weight convention matches `Dense`: `weight[post, pre]`.

    `a_minus` larger than `a_plus` is the usual choice: symmetric rates run away, because a
    strengthened synapse makes its postsynaptic neuron fire more, which strengthens it further.
    """

    tau_pre: float = eqx.field(static=True)
    tau_post: float = eqx.field(static=True)
    a_plus: float = eqx.field(static=True)
    a_minus: float = eqx.field(static=True)
    w_min: float = eqx.field(static=True)
    w_max: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
    ):
        if w_min >= w_max:
            raise ValueError(f"w_min ({w_min}) must be below w_max ({w_max})")
        self.tau_pre, self.tau_post = tau_pre, tau_post
        self.a_plus, self.a_minus = a_plus, a_minus
        self.w_min, self.w_max = w_min, w_max
        self.dt = dt

    def init_state(self, batch: int, n_pre: int, n_post: int) -> STDPState:
        return STDPState(pre=jnp.zeros((batch, n_pre)), post=jnp.zeros((batch, n_post)))

    def step(
        self, state: STDPState, pre_spikes: Array, post_spikes: Array
    ) -> tuple[STDPState, Array]:
        """One timestep. Returns the new traces and this step's weight delta `[post, pre]`."""
        decay_pre = jnp.exp(-self.dt / self.tau_pre)
        decay_post = jnp.exp(-self.dt / self.tau_post)

        # Traces carried in from t-1: a pre and post spike in the same step are simultaneous,
        # not causal, so neither should see the other's contribution.
        potentiation = self.a_plus * jnp.einsum("bi,bj->ij", post_spikes, state.pre)
        depression = self.a_minus * jnp.einsum("bi,bj->ij", state.post, pre_spikes)

        new_state = STDPState(
            pre=decay_pre * state.pre + pre_spikes,
            post=decay_post * state.post + post_spikes,
        )
        return new_state, potentiation - depression

    def __call__(
        self,
        weight: Float[Array, "post pre"],
        pre_spikes: Float[Array, "T B pre"],
        post_spikes: Float[Array, "T B post"],
        *,
        state: STDPState | None = None,
        learning_rate: float = 1.0,
    ) -> tuple[Array, STDPState]:
        """Apply STDP over a whole spike sequence. Returns `(new_weight, final_traces)`."""
        if pre_spikes.shape[0] != post_spikes.shape[0]:
            raise ValueError(
                f"pre and post must cover the same timesteps, got {pre_spikes.shape[0]} "
                f"and {post_spikes.shape[0]}"
            )
        if state is None:
            state = self.init_state(pre_spikes.shape[1], pre_spikes.shape[2], post_spikes.shape[2])

        def scan_step(carry, inputs):
            traces, accumulated = carry
            pre_t, post_t = inputs
            traces, delta = self.step(traces, pre_t, post_t)
            return (traces, accumulated + delta), None

        (final_traces, total), _ = jax.lax.scan(
            scan_step, (state, jnp.zeros_like(weight)), (pre_spikes, post_spikes)
        )
        updated = jnp.clip(weight + learning_rate * total, self.w_min, self.w_max)
        return updated, final_traces


def stdp_window(
    delta_t: Array,
    *,
    tau_pre: float = 20.0,
    tau_post: float = 20.0,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
) -> Array:
    """The classic STDP curve: weight change against `t_post - t_pre`.

    Positive `delta_t` means post fired after pre (causal, strengthens); negative means the
    reverse (weakens). Useful for plotting and for checking a configuration does what you
    think before running it on a network.
    """
    return jnp.where(
        delta_t > 0,
        a_plus * jnp.exp(-jnp.abs(delta_t) / tau_pre),
        jnp.where(delta_t < 0, -a_minus * jnp.exp(-jnp.abs(delta_t) / tau_post), 0.0),
    )


# Tsodyks-Pawelzik-Markram (1998) parameter sets, as reproduced in the Brian2 example of that
# paper. Time constants are in milliseconds, so pair them with dt=1.0 for a 1 ms step.
MARKRAM_PRESETS: dict[str, tuple[float, float, float]] = {
    #                  U     tau_d (recovery)  tau_f (facilitation)
    "depressing": (0.60, 800.0, 0.0),
    "facilitating": (0.03, 130.0, 530.0),
    # Markram, Wang & Tsodyks (1998) classes between pyramidal cells and interneurons.
    "F1_facilitating": (0.16, 45.0, 376.0),
    "F2_depressing": (0.25, 706.0, 21.0),
    "F3_mixed": (0.32, 144.0, 62.0),
}


class MarkramState(eqx.Module):
    u: Float[Array, "..."]  # utilization / release probability (facilitation)
    x: Float[Array, "..."]  # fraction of resources available (depression)


class TsodyksMarkram(eqx.Module):
    """Tsodyks-Markram short-term plasticity: synapses that fatigue and facilitate.

    This is not a learning rule -- nothing here is remembered across a stimulus. It is a
    *dynamic synapse*: efficacy changes over tens to hundreds of milliseconds because vesicles
    deplete and presynaptic calcium accumulates, then relaxes back, so the synapse transmits a
    spike *train* differently depending on its rate and history.

    Two variables per presynaptic neuron:

        u -- utilization, the fraction of available resources released per spike. Decays to 0
             with `tau_f`; each spike pushes it up by ``U*(1-u)``. This is *facilitation*.
        x -- resources available. Recovers toward 1 with `tau_d`; each spike consumes ``u*x``.
             This is *depression*.

    The transmitted amplitude is ``u*x`` rather than a binary 1, so the output is graded. Place
    it directly after a spiking layer and before the weights it modulates::

        jp.Sequential(jp.Dense(...), jp.LIF(...), jp.TsodyksMarkram.preset("depressing"),
                      jp.Dense(...), ...)

    With `tau_f = 0` facilitation is off and the synapse purely depresses, which is the
    original 1997 model.

    Both recurrences are affine given the spike train, so this parallelizes over time.
    """

    U: float = eqx.field(static=True)
    tau_d: float = eqx.field(static=True)
    tau_f: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(
        self, U: float = 0.6, tau_d: float = 800.0, tau_f: float = 0.0, *, dt: float = 1.0
    ):
        if not 0.0 < U <= 1.0:
            raise ValueError(f"U is a release probability and must be in (0, 1], got {U}")
        if tau_d <= 0.0:
            raise ValueError(f"tau_d must be positive, got {tau_d}")
        if tau_f < 0.0:
            raise ValueError(f"tau_f must be non-negative (0 disables facilitation), got {tau_f}")
        self.U, self.tau_d, self.tau_f, self.dt = U, tau_d, tau_f, dt

    @classmethod
    def preset(cls, name: str, *, dt: float = 1.0) -> TsodyksMarkram:
        """Construct from a published parameter set, e.g. ``preset("facilitating")``."""
        if name not in MARKRAM_PRESETS:
            raise ValueError(f"unknown preset {name!r}; available: {sorted(MARKRAM_PRESETS)}")
        u, tau_d, tau_f = MARKRAM_PRESETS[name]
        return cls(u, tau_d, tau_f, dt=dt)

    @property
    def decay_d(self) -> float:
        # math, not jnp: jnp on a static float creates a tracer that cannot be converted back
        # to float inside a jitted scan.
        return math.exp(-self.dt / self.tau_d)

    @property
    def decay_f(self) -> float:
        # tau_f == 0 means no facilitation: u falls straight back to 0 between spikes.
        return 0.0 if self.tau_f == 0.0 else math.exp(-self.dt / self.tau_f)

    def init_state(self, input_shape: tuple[int, ...]) -> MarkramState:
        return MarkramState(
            u=jnp.zeros(input_shape, dtype=jnp.float32),
            x=jnp.ones(input_shape, dtype=jnp.float32),
        )

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def __call__(self, state: MarkramState, spikes: Array) -> tuple[MarkramState, Array]:
        s = spikes.astype(jnp.float32)
        # Facilitation: decay, then a spike raises u toward 1 by U*(1-u).
        u = state.u * self.decay_f
        u = u + self.U * (1.0 - u) * s
        # Depression: resources recover toward 1, then a spike consumes u*x of them.
        x = state.x * self.decay_d + (1.0 - self.decay_d)
        released = u * x * s
        return MarkramState(u=u, x=x - released), released

    def parallel_apply(self, state: MarkramState, spikes: Array) -> tuple[MarkramState, Array]:
        """Both variables are affine in the spike train, so each is one associative scan."""
        from .parallel import scan_linear_recurrence

        s = spikes.astype(jnp.float32)
        # u[t] = decay_f*(1 - U*s[t]) * u[t-1] + U*s[t]
        u = scan_linear_recurrence(self.decay_f * (1.0 - self.U * s), self.U * s, state.u)
        # x_pre[t] = decay_d*x[t-1] + (1-decay_d);  x[t] = x_pre[t] * (1 - u[t]*s[t])
        keep = 1.0 - u * s
        x = scan_linear_recurrence(self.decay_d * keep, (1.0 - self.decay_d) * keep, state.x)
        # The released amount uses x *before* this step's consumption.
        x_prev = jnp.concatenate([state.x[None], x[:-1]], axis=0)
        x_pre = x_prev * self.decay_d + (1.0 - self.decay_d)
        released = u * x_pre * s
        return MarkramState(u=u[-1], x=x[-1]), released


class DopamineState(eqx.Module):
    pre: Float[Array, "B pre"]
    post: Float[Array, "B post"]
    eligibility: Float[Array, "post pre"]
    dopamine: Float[Array, ""]


class DopamineSTDP(eqx.Module):
    """Reward-modulated STDP (Izhikevich 2007), a three-factor rule.

    Plain STDP changes a weight the moment two spikes coincide, which cannot explain learning
    from *delayed* reward: by the time a reward arrives seconds later, the responsible spike
    pair is long gone. Izhikevich's answer is to insert a slow variable between them:

        eligibility c: STDP writes into c, not into the weight.  c decays with tau_c (~1 s).
        dopamine d:    reward raises d.                          d decays with tau_d (~200 ms).
        weight:        dw/dt = c * d.

    The weight moves only where eligibility and dopamine *overlap*. A spike pair leaves a tag
    that persists for about a second; if reward arrives inside that window the synapse is
    reinforced, otherwise the tag decays. Random firing during the wait does not wash it out,
    because STDP's own window is only ~20 ms wide.

    Defaults are the paper's, in milliseconds, so use `dt=1.0`.
    """

    tau_pre: float = eqx.field(static=True)
    tau_post: float = eqx.field(static=True)
    tau_c: float = eqx.field(static=True)
    tau_dopamine: float = eqx.field(static=True)
    a_plus: float = eqx.field(static=True)
    a_minus: float = eqx.field(static=True)
    dopamine_per_reward: float = eqx.field(static=True)
    w_min: float = eqx.field(static=True)
    w_max: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        tau_c: float = 1000.0,
        tau_dopamine: float = 200.0,
        a_plus: float = 0.01,
        a_minus: float = 0.0105,  # paper uses A- = 1.05 * A+
        dopamine_per_reward: float = 5e-3,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
    ):
        if w_min >= w_max:
            raise ValueError(f"w_min ({w_min}) must be below w_max ({w_max})")
        self.tau_pre, self.tau_post = tau_pre, tau_post
        self.tau_c, self.tau_dopamine = tau_c, tau_dopamine
        self.a_plus, self.a_minus = a_plus, a_minus
        self.dopamine_per_reward = dopamine_per_reward
        self.w_min, self.w_max, self.dt = w_min, w_max, dt

    def init_state(self, batch: int, n_pre: int, n_post: int) -> DopamineState:
        return DopamineState(
            pre=jnp.zeros((batch, n_pre)),
            post=jnp.zeros((batch, n_post)),
            eligibility=jnp.zeros((n_post, n_pre)),
            dopamine=jnp.zeros(()),
        )

    def step(
        self, state: DopamineState, pre_spikes: Array, post_spikes: Array, reward: Array
    ) -> tuple[DopamineState, Array]:
        """One timestep. Returns new state and this step's weight delta."""
        potentiation = self.a_plus * jnp.einsum("bi,bj->ij", post_spikes, state.pre)
        depression = self.a_minus * jnp.einsum("bi,bj->ij", state.post, pre_spikes)

        # STDP writes into the eligibility trace, never directly into the weight.
        eligibility = state.eligibility * math.exp(-self.dt / self.tau_c)
        eligibility = eligibility + (potentiation - depression)
        dopamine = state.dopamine * math.exp(-self.dt / self.tau_dopamine)
        dopamine = dopamine + self.dopamine_per_reward * jnp.asarray(reward, jnp.float32)

        new_state = DopamineState(
            pre=state.pre * math.exp(-self.dt / self.tau_pre) + pre_spikes,
            post=state.post * math.exp(-self.dt / self.tau_post) + post_spikes,
            eligibility=eligibility,
            dopamine=dopamine,
        )
        return new_state, eligibility * dopamine * self.dt

    def __call__(
        self,
        weight: Float[Array, "post pre"],
        pre_spikes: Float[Array, "T B pre"],
        post_spikes: Float[Array, "T B post"],
        reward: Float[Array, " T"],
        *,
        state: DopamineState | None = None,
        learning_rate: float = 1.0,
    ) -> tuple[Array, DopamineState]:
        """Run the rule over a sequence. `reward` is a per-timestep scalar signal."""
        if not (pre_spikes.shape[0] == post_spikes.shape[0] == reward.shape[0]):
            raise ValueError(
                "pre, post and reward must cover the same timesteps, got "
                f"{pre_spikes.shape[0]}, {post_spikes.shape[0]}, {reward.shape[0]}"
            )
        if state is None:
            state = self.init_state(pre_spikes.shape[1], pre_spikes.shape[2], post_spikes.shape[2])

        def scan_step(carry, inputs):
            traces, accumulated = carry
            pre_t, post_t, reward_t = inputs
            traces, delta = self.step(traces, pre_t, post_t, reward_t)
            return (traces, accumulated + delta), None

        (final, total), _ = jax.lax.scan(
            scan_step, (state, jnp.zeros_like(weight)), (pre_spikes, post_spikes, reward)
        )
        return jnp.clip(weight + learning_rate * total, self.w_min, self.w_max), final


__all__ = [
    "MARKRAM_PRESETS",
    "STDP",
    "DopamineSTDP",
    "DopamineState",
    "MarkramState",
    "STDPState",
    "TsodyksMarkram",
    "stdp_window",
]
