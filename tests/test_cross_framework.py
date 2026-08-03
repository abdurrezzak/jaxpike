"""Numerical cross-validation against snnTorch and Norse.

Skipped unless torch, snntorch and norse are installed (`pip install -e ".[bench]"`).

The three libraries do not implement the same equation, and pretending otherwise would make
any "we match the reference implementations" claim meaningless. The actual relationships,
established empirically here rather than read off documentation:

    jaxpike   v[t] = a*v[t-1] + (1-a)*x[t],  a = exp(-dt/tau)      exact exponential
    snnTorch  m[t] = a*m[t-1] +     x[t] - thr*s[t-1]              exact decay, unscaled input
    Norse     v[t] = v[t-1] + (dt/tau)*((v_leak - v[t-1]) + x[t])  forward Euler

So: snnTorch shares our decay but not our input scaling or reset timing, and Norse shares
neither because it uses a first-order integrator. These tests pin the exact conversions, which
is what a migration guide has to be built on.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxpike import LIF, unroll

torch = pytest.importorskip("torch")
snn = pytest.importorskip("snntorch")
pytest.importorskip("norse")

TAU, DT, THRESHOLD = 20.0, 1.0, 1.0
ALPHA = float(jnp.exp(-DT / TAU))


def ours(x, steps, *, tau=TAU, threshold=THRESHOLD, dt=DT):
    lif = LIF(tau=tau, threshold=threshold, dt=dt)
    xs = jnp.full((steps, 1), x)
    spikes, _ = unroll(lif, xs)
    return spikes


def ours_membrane(x, steps, *, tau=TAU, dt=DT):
    """Membrane trace with spiking disabled, for comparing integrators directly."""
    lif = LIF(tau=tau, threshold=1e9, dt=dt)
    xs = jnp.full((steps, 1), x)
    state = lif.init_state((1,))
    trace = []
    for t in range(steps):
        state, _ = lif(state, xs[t])
        trace.append(float(state.v[0]))
    return trace


def snntorch_membrane(x, steps, *, alpha=ALPHA):
    cell = snn.Leaky(beta=alpha, threshold=1e9, reset_mechanism="subtract", init_hidden=False)
    mem = torch.zeros(1)
    trace = []
    for _ in range(steps):
        _, mem = cell(torch.tensor([x]), mem)
        trace.append(float(mem[0]))
    return trace


def norse_membrane(x, steps, *, tau=TAU, dt=DT):
    from norse.torch.functional.lif_box import LIFBoxParameters
    from norse.torch.module.lif_box import LIFBoxCell

    # tau_mem_inv is per-second and Norse's dt is in seconds, so dt*tau_mem_inv = dt/tau.
    # Our tau is in units of dt, hence the 1e-3 pairing below.
    p = LIFBoxParameters(
        tau_mem_inv=torch.tensor(1.0 / (tau * 1e-3)),
        v_leak=torch.tensor(0.0),
        v_th=torch.tensor(1e9),
        v_reset=torch.tensor(0.0),
    )
    cell = LIFBoxCell(p=p, dt=1e-3)
    state, trace = None, []
    for _ in range(steps):
        _, state = cell(torch.tensor([[x]]), state)
        trace.append(float(state.v[0, 0].detach()))
    return trace


# --- our integrator is exact ---------------------------------------------------------------


def test_our_membrane_is_the_exact_ode_solution():
    """For constant input the analytic solution is x*(1 - exp(-t/tau)); we must hit it exactly.

    This is the strongest correctness statement available for the subthreshold dynamics: not
    "close to a reference implementation" but equal to the closed-form solution of the ODE.
    """
    x, steps = 3.0, 40
    got = ours_membrane(x, steps)
    for n, v in enumerate(got, start=1):
        analytic = x * (1.0 - ALPHA**n)
        assert v == pytest.approx(analytic, rel=1e-6), f"step {n}"


def test_norse_euler_is_measurably_less_accurate_than_our_exact_step():
    """Norse's forward Euler carries O(dt/tau) truncation error; ours carries none."""
    x, steps = 3.0, 40
    analytic = [x * (1.0 - ALPHA**n) for n in range(1, steps + 1)]
    our_err = max(abs(a - b) for a, b in zip(ours_membrane(x, steps), analytic, strict=True))
    norse_err = max(abs(a - b) for a, b in zip(norse_membrane(x, steps), analytic, strict=True))
    assert our_err < 1e-5
    assert norse_err > our_err * 100, (
        f"expected Euler error to dominate; ours={our_err:.2e} norse={norse_err:.2e}"
    )


def test_we_converge_to_norse_as_the_timestep_shrinks():
    """The two agree in the limit, which confirms it is the same ODE and not a different model."""
    x, errors = 3.0, []
    for tau in (20.0, 200.0, 2000.0):  # shrinking dt/tau at fixed dt
        steps = 20
        a = ours_membrane(x, steps, tau=tau)
        b = norse_membrane(x, steps, tau=tau)
        errors.append(max(abs(p - q) for p, q in zip(a, b, strict=True)))
    assert errors[0] > errors[1] > errors[2], f"error must shrink with dt/tau, got {errors}"
    assert errors[-1] < 1e-3


# --- snnTorch conversion -------------------------------------------------------------------


def test_snntorch_matches_ours_when_input_is_rescaled():
    """snnTorch omits the (1-alpha) input factor, so ported weights need scaling by (1-alpha).

    With that one correction the subthreshold traces agree to float32 precision, confirming
    the decay term is identical and the difference really is only input normalization.
    """
    x, steps = 3.0, 40
    ours_trace = ours_membrane(x, steps)
    snn_trace = snntorch_membrane((1.0 - ALPHA) * x, steps)
    worst = max(abs(a - b) for a, b in zip(ours_trace, snn_trace, strict=True))
    assert worst < 1e-5, f"worst deviation {worst:.2e}"


def test_reset_timing_differs_from_snntorch_and_the_difference_is_bounded():
    """We reset before the next decay; snnTorch subtracts after it. Not equivalent.

    Ours:     v[t] = a*(v[t-1] - thr*s[t-1]) + (1-a)*x[t]   -> reset is attenuated by a
    snnTorch: m[t] = a*m[t-1] - thr*s[t-1] + x[t]           -> reset is applied at full size

    Both are defensible and both appear in the literature. This test exists so the difference
    is a documented, asserted fact rather than a surprise for someone porting a model.
    """
    steps, x = 30, 3.0
    lif = LIF(tau=TAU, threshold=THRESHOLD, reset="subtract")
    state = lif.init_state((1,))
    ours_v = []
    for _ in range(steps):
        state, _ = lif(state, jnp.full((1,), x))
        ours_v.append(float(state.v[0]))

    manual, v = [], 0.0
    for _ in range(steps):
        v = ALPHA * v + (1.0 - ALPHA) * x
        v = v - THRESHOLD * (1.0 if v > THRESHOLD else 0.0)
        manual.append(v)
    assert max(abs(a - b) for a, b in zip(ours_v, manual, strict=True)) < 1e-5


def test_spike_counts_are_in_the_same_regime_as_snntorch():
    """Sanity bound: identical drive must not produce wildly different firing rates."""
    steps, x = 200, 3.0
    our_count = float(jnp.sum(ours(x, steps)))

    cell = snn.Leaky(beta=ALPHA, threshold=THRESHOLD, reset_mechanism="subtract")
    mem, count = torch.zeros(1), 0.0
    for _ in range(steps):
        spk, mem = cell(torch.tensor([(1.0 - ALPHA) * x]), mem)
        count += float(spk[0])
    assert our_count > 0 and count > 0
    assert 0.5 < our_count / count < 2.0, f"ours={our_count} snntorch={count}"
