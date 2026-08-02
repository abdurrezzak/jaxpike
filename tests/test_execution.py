"""Execution engine tests.

`unroll_checkpointed` must be indistinguishable from `unroll` in every observable way --
outputs, final state, and gradients. It only differs in how much memory it uses getting
there. Everything Phase 2 adds (fused kernels, parallel-in-time) must clear the same bar,
so these tests are the template for validating those backends.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxpike import LIF, Dense, Sequential, unroll, unroll_checkpointed
from jaxpike.execution import _best_chunk


def net(key=None):
    k1, k2 = jax.random.split(key if key is not None else jax.random.key(0))
    return Sequential(Dense(8, 16, key=k1), LIF(tau=20.0), Dense(16, 4, key=k2), LIF(tau=20.0))


def inputs(t=100, batch=3, features=8, seed=0):
    return jax.random.normal(jax.random.key(seed), (t, batch, features)) * 2.0


def test_checkpointed_matches_naive_outputs_and_state():
    model, xs = net(), inputs()
    a, state_a = unroll(model, xs)
    b, state_b = unroll_checkpointed(model, xs)
    assert jnp.array_equal(a, b)
    assert jnp.allclose(state_a[3].v, state_b[3].v, atol=1e-6)


def test_checkpointed_matches_naive_gradients():
    model, xs = net(), inputs()

    def loss(runner):
        return lambda m, x: jnp.mean(runner(m, x)[0])

    g_naive = eqx.filter_grad(loss(unroll))(model, xs)
    g_ckpt = eqx.filter_grad(loss(unroll_checkpointed))(model, xs)

    leaves_n = jax.tree.leaves(eqx.filter(g_naive, eqx.is_inexact_array))
    leaves_c = jax.tree.leaves(eqx.filter(g_ckpt, eqx.is_inexact_array))
    assert len(leaves_n) == len(leaves_c)
    for a, b in zip(leaves_n, leaves_c, strict=True):
        assert jnp.allclose(a, b, atol=1e-5), "rematerialization must not change gradients"
    assert any(jnp.any(leaf != 0) for leaf in leaves_n), "gradients were trivially zero"


@pytest.mark.parametrize("chunk", [1, 2, 5, 10, 25, 50])
def test_any_dividing_chunk_size_gives_identical_results(chunk):
    model, xs = net(), inputs(t=50)
    expected, _ = unroll(model, xs)
    got, _ = unroll_checkpointed(model, xs, chunk_size=chunk)
    assert jnp.array_equal(expected, got)


def test_non_dividing_chunk_size_is_rejected():
    # Padding would silently advance the recurrence and corrupt the final state, so this
    # must fail loudly rather than quietly return something almost right.
    model, xs = net(), inputs(t=50)
    with pytest.raises(ValueError, match="must divide"):
        unroll_checkpointed(model, xs, chunk_size=7)


def test_checkpointed_supports_streaming_continuation():
    model, xs = net(), inputs(t=100)
    full, full_state = unroll(model, xs)
    first, mid = unroll_checkpointed(model, xs[:60])
    second, end = unroll_checkpointed(model, xs[60:], mid)
    assert jnp.allclose(jnp.concatenate([first, second]), full, atol=1e-6)
    assert jnp.allclose(end[3].v, full_state[3].v, atol=1e-6)


@pytest.mark.parametrize("t", [1, 4, 16, 36, 100, 101, 997])
def test_best_chunk_always_divides_and_is_near_sqrt(t):
    chunk = _best_chunk(t)
    assert t % chunk == 0
    assert 1 <= chunk <= t


def test_best_chunk_picks_sqrt_for_perfect_squares():
    assert _best_chunk(100) == 10
    assert _best_chunk(2500) == 50


def test_checkpointed_uses_less_scratch_memory():
    # The whole point, asserted rather than assumed. Measured through XLA's own allocation
    # plan, which is deterministic and works on CPU.
    model = net()
    xs = inputs(t=900)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def temp_bytes(runner):
        def loss(p, x):
            return jnp.mean(runner(eqx.combine(p, static), x)[0])

        compiled = jax.jit(jax.grad(loss)).lower(params, xs).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    naive, ckpt = temp_bytes(unroll), temp_bytes(unroll_checkpointed)
    assert ckpt * 5 < naive, f"expected >=5x saving, got {naive / max(ckpt, 1):.1f}x"
