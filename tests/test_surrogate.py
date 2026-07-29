"""Surrogate gradient tests.

The relaxation-based design means the surrogate derivative *can* be finite-difference
checked, which is the point of building it this way -- a hand-written surrogate VJP can
silently disagree with its own forward relaxation and no test would catch it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxpike import ATan, Boxcar, FastSigmoid, Triangular

ALL_SURROGATES = [FastSigmoid(), ATan(), Triangular(), Boxcar()]
IDS = [type(s).__name__ for s in ALL_SURROGATES]


@pytest.mark.parametrize("sg", ALL_SURROGATES, ids=IDS)
def test_forward_is_exact_heaviside(sg):
    v = jnp.linspace(-3.0, 3.0, 101)
    out = sg(v)
    assert jnp.all((out == 0.0) | (out == 1.0)), "spikes must be exactly binary"
    assert jnp.array_equal(out, (v > 0).astype(v.dtype))


@pytest.mark.parametrize("sg", ALL_SURROGATES, ids=IDS)
def test_gradient_matches_finite_difference_of_relaxation(sg):
    # Away from kinks in the relaxation, autodiff and central differences must agree.
    v = jnp.array([-2.0, -0.7, -0.13, 0.13, 0.7, 2.0])
    eps = 1e-3
    analytic = jax.vmap(jax.grad(lambda z: sg(z).sum()))(v)
    numeric = (sg.relaxation(v + eps) - sg.relaxation(v - eps)) / (2 * eps)
    assert jnp.allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("sg", ALL_SURROGATES, ids=IDS)
def test_gradient_is_peaked_at_threshold(sg):
    g = jax.vmap(jax.grad(lambda z: sg(z).sum()))
    at_zero = g(jnp.array([0.0]))[0]
    far = g(jnp.array([5.0]))[0]
    assert at_zero > 0.0, "no gradient at threshold means no learning signal"
    assert at_zero > far


@pytest.mark.parametrize("sg", ALL_SURROGATES, ids=IDS)
def test_gradient_is_finite_over_wide_range(sg):
    v = jnp.linspace(-1e4, 1e4, 2001)
    g = jax.vmap(jax.grad(lambda z: sg(z).sum()))(v)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("sg", [Triangular(width=1.0), Boxcar(width=1.0)], ids=["tri", "box"])
def test_compact_support_surrogates_vanish_outside_window(sg):
    g = jax.vmap(jax.grad(lambda z: sg(z).sum()))
    assert jnp.all(g(jnp.array([-5.0, -2.0, 2.0, 5.0])) == 0.0)
    assert jnp.all(g(jnp.array([-0.2, 0.0, 0.2])) > 0.0)


def test_slope_scales_gradient_magnitude():
    g = lambda sg: jax.grad(lambda z: sg(z).sum())(jnp.array(0.3))  # noqa: E731
    assert g(FastSigmoid(slope=5.0)) > g(FastSigmoid(slope=50.0))


@pytest.mark.parametrize("sg", ALL_SURROGATES, ids=IDS)
def test_jit_and_vmap_agree_with_eager(sg):
    v = jnp.linspace(-2.0, 2.0, 32).reshape(4, 8)
    assert jnp.array_equal(sg(v), jax.jit(sg)(v))
    assert jnp.array_equal(sg(v), jax.vmap(sg)(v))
