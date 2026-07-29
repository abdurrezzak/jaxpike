"""Neuron model tests.

A note on what is *not* tested here, because it matters. The spiking path cannot be
finite-difference checked: the forward pass is piecewise constant, so its true derivative is
zero almost everywhere and infinite at threshold crossings. Any finite-difference comparison
against a surrogate gradient will disagree, and that disagreement is the entire design of
surrogate gradient learning rather than a bug.

So the neuron gradients are pinned three other ways: the surrogate itself is
finite-difference checked (`test_surrogate.py`), the smooth membrane path is checked here with
spikes held fixed, and the full spiking path is validated for structural correctness --
finite, non-zero, flowing to every parameter including the learnable time constant.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxpike import ALIF, LIF, Dense, Sequential, spike_count, unroll
from jaxpike.neurons import STATE_DTYPE

T, B, N = 60, 4, 16


def constant_drive(value, t=T, shape=(B, N)):
    return jnp.full((t, *shape), value)


# --- forward-pass invariants -------------------------------------------------------------


@pytest.mark.parametrize("reset", ["subtract", "zero"])
def test_output_is_strictly_binary(reset):
    lif = LIF(tau=20.0, reset=reset)
    spikes, _ = unroll(lif, constant_drive(3.0))
    assert jnp.all((spikes == 0.0) | (spikes == 1.0))


def test_subthreshold_drive_never_spikes():
    # v converges to the input value, so a drive below threshold can never cross it.
    lif = LIF(tau=20.0, threshold=1.0)
    spikes, state = unroll(lif, constant_drive(0.5))
    assert spike_count(spikes).sum() == 0
    assert jnp.all(state.v < 1.0)


def test_suprathreshold_drive_spikes():
    lif = LIF(tau=20.0, threshold=1.0)
    spikes, _ = unroll(lif, constant_drive(5.0))
    assert spike_count(spikes).min() > 0


def test_zero_reset_clamps_membrane_and_subtract_retains_overshoot():
    # Input is scaled by (1 - alpha), so a single step injects only ~5% of the drive at
    # tau=20. The drive has to be large for one step to overshoot threshold at all.
    x = jnp.full((1,), 100.0)
    sub = LIF(tau=20.0, reset="subtract")
    zero = LIF(tau=20.0, reset="zero")
    sub_state, sub_spike = sub(sub.init_state((1,)), x)
    zero_state, zero_spike = zero(zero.init_state((1,)), x)

    assert sub_spike.item() == 1.0 and zero_spike.item() == 1.0, "drive must actually fire"
    assert zero_state.v.item() == pytest.approx(0.0, abs=1e-6)
    assert sub_state.v.item() > 0.0, "subtract reset must retain the suprathreshold remainder"


def test_membrane_stays_bounded_under_bounded_drive():
    lif = LIF(tau=20.0)
    xs = jax.random.uniform(jax.random.key(0), (500, B, N), minval=0.0, maxval=2.0)
    _, state = unroll(lif, xs)
    assert jnp.all(jnp.isfinite(state.v))
    assert jnp.all(jnp.abs(state.v) < 10.0)


def test_state_is_float32_even_for_bf16_input():
    lif = LIF(tau=20.0)
    xs = constant_drive(3.0).astype(jnp.bfloat16)
    _, state = unroll(lif, xs)
    assert state.v.dtype == STATE_DTYPE, "membrane must accumulate in fp32 to avoid drift"


def test_rejects_tau_below_dt():
    with pytest.raises(ValueError, match="tau must exceed dt"):
        LIF(tau=0.5, dt=1.0)


def test_rejects_unknown_reset():
    with pytest.raises(ValueError, match="reset must be"):
        LIF(tau=20.0, reset="magic")


# --- transformation invariants -----------------------------------------------------------


def test_jit_matches_eager():
    lif = LIF(tau=20.0)
    xs = constant_drive(3.0)
    eager, _ = unroll(lif, xs)
    compiled, _ = eqx.filter_jit(unroll)(lif, xs)
    assert jnp.array_equal(eager, compiled)


def test_deterministic_across_repeated_calls():
    lif = LIF(tau=20.0)
    xs = jax.random.normal(jax.random.key(1), (T, B, N))
    a, _ = unroll(lif, xs)
    b, _ = unroll(lif, xs)
    assert jnp.array_equal(a, b)


def test_batching_is_independent_across_examples():
    # Neurons are elementwise, so example i of a batched run must equal a solo run of example i.
    lif = LIF(tau=20.0)
    xs = jax.random.normal(jax.random.key(2), (T, B, N)) * 2.0
    batched, _ = unroll(lif, xs)
    solo, _ = unroll(lif, xs[:, 1])
    assert jnp.allclose(batched[:, 1], solo, atol=1e-6)


def test_chunked_run_equals_single_run():
    # State continuity: the property that makes truncated BPTT and streaming inference correct.
    lif = LIF(tau=20.0)
    xs = jax.random.normal(jax.random.key(3), (T, B, N)) * 2.0
    full, full_state = unroll(lif, xs)
    first, mid = unroll(lif, xs[:25])
    second, end_state = unroll(lif, xs[25:], mid)
    assert jnp.allclose(jnp.concatenate([first, second]), full, atol=1e-6)
    assert jnp.allclose(end_state.v, full_state.v, atol=1e-6)


# --- gradient behaviour ------------------------------------------------------------------


def _loss(net, xs, state=None):
    spikes, _ = unroll(net, xs, state)
    return jnp.mean(spikes)


def test_gradients_reach_every_parameter_including_tau():
    key = jax.random.key(4)
    net = Sequential(Dense(N, 32, key=key), LIF(tau=20.0), Dense(32, 8, key=key), LIF(tau=15.0))
    xs = jax.random.normal(jax.random.key(5), (T, B, N)) * 3.0
    grads = eqx.filter_grad(_loss)(net, xs)

    leaves = [g for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array)) if g.size]
    assert leaves, "no differentiable leaves found"
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))
    assert any(jnp.any(g != 0.0) for g in leaves), "all gradients vanished"

    tau_grads = [grads.layers[i].log_tau for i in (1, 3)]
    for g in tau_grads:
        assert jnp.all(jnp.isfinite(g))
    assert any(jnp.any(g != 0.0) for g in tau_grads), "time constants received no gradient"


def test_membrane_path_gradient_matches_finite_difference():
    # With spikes frozen the neuron is a smooth linear filter, so real derivatives exist here.
    lif = LIF(tau=20.0, threshold=1e9)  # unreachable threshold => no spikes, no kinks

    def membrane(scale):
        _, state = unroll(lif, constant_drive(1.0) * scale)
        return jnp.sum(state.v)

    analytic = jax.grad(membrane)(2.0)
    eps = 1e-3
    numeric = (membrane(2.0 + eps) - membrane(2.0 - eps)) / (2 * eps)
    assert analytic == pytest.approx(numeric, rel=1e-3)


def test_gradient_does_not_explode_over_long_sequences():
    lif = LIF(tau=20.0)
    xs = jnp.full((2000, 2, 8), 1.2)
    g = eqx.filter_grad(_loss)(lif, xs)
    assert jnp.all(jnp.isfinite(g.log_tau))
    assert jnp.abs(g.log_tau).max() < 1e4


# --- ALIF --------------------------------------------------------------------------------


def test_alif_adaptation_suppresses_firing_relative_to_lif():
    xs = constant_drive(2.0, t=500)
    lif_spikes, _ = unroll(LIF(tau=20.0), xs)
    alif_spikes, _ = unroll(ALIF(tau=20.0, tau_a=200.0, beta=1.8), xs)
    assert spike_count(alif_spikes).sum() < spike_count(lif_spikes).sum()


def test_alif_carries_two_state_variables():
    alif = ALIF(tau=20.0)
    _, state = unroll(alif, constant_drive(3.0))
    assert jnp.all(jnp.isfinite(state.v))
    assert jnp.all(state.a >= 0.0), "adaptation integrates non-negative spike counts"
    assert jnp.any(state.a > 0.0)
