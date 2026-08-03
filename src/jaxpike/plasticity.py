"""Spike-timing-dependent plasticity.

Everything else in this library trains by backpropagation through time with surrogate
gradients: a global loss, gradients pushed backwards through the whole unrolled network.
STDP is a different animal entirely and it is worth being clear about the difference before
using it.

STDP is **local and unsupervised**. A synapse changes strength based only on the relative
timing of the spikes at its two ends: if the presynaptic neuron fires shortly *before* the
postsynaptic one, the synapse strengthens; if shortly *after*, it weakens. No loss function,
no gradients, no backward pass, and no information from anywhere else in the network. That
locality is why neuromorphic chips can implement it in hardware and why it is the standard
model of biological learning.

The trade is real: STDP has no notion of a task. It extracts correlation structure from
input statistics, which is useful for unsupervised feature learning and as a biologically
plausible mechanism to study, but on its own it will not train a classifier the way BPTT
will. The usual practical recipes are STDP-pretraining followed by a supervised readout, or
a three-factor rule where a reward signal gates the update.

The implementation is the standard pair-based rule with exponential eligibility traces:

    trace_pre[t]  = exp(-dt/tau_pre) * trace_pre[t-1]   + pre_spikes[t]
    trace_post[t] = exp(-dt/tau_post) * trace_post[t-1] + post_spikes[t]

    dw += a_plus  * outer(post_spikes[t], trace_pre[t-1])     # post after pre -> strengthen
    dw -= a_minus * outer(trace_post[t-1], pre_spikes[t])     # pre after post -> weaken

Traces from `t-1` are used deliberately: a pre and post spike in the same timestep are
simultaneous, not causally ordered, and should not potentiate.
"""

from __future__ import annotations

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

    `a_minus` larger than `a_plus` is the usual choice. Symmetric rates tend to run away,
    because a strengthened synapse makes its postsynaptic neuron fire more, which strengthens
    it further; slight depression bias plus the `w_min`/`w_max` clamp keeps that in check.
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


__all__ = ["STDP", "STDPState", "stdp_window"]
