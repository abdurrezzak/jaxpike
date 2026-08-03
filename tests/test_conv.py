"""Convolution, pooling and flatten tests.

Layout throughout is NHWC: `(time, batch, height, width, channels)`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxpike import (
    Conv2d,
    Dense,
    Flatten,
    LinearLIF,
    Pool2d,
    Sequential,
    lif_gain,
    unroll,
    unroll_parallel,
)

TAU = 20.0
GAIN = lif_gain(TAU)


def convnet(key=None, threshold=0.2, gain=GAIN):
    # gain is not optional here: without it this network is silent by layer three. See
    # jaxpike.init.lif_gain and test_init.py.
    k = jax.random.split(key if key is not None else jax.random.key(0), 3)
    return Sequential(
        Conv2d(2, 8, 3, key=k[0], gain=gain),
        LinearLIF(tau=TAU, threshold=threshold),
        Pool2d(2),
        Conv2d(8, 16, 3, key=k[1], gain=gain),
        LinearLIF(tau=TAU, threshold=threshold),
        Pool2d(2),
        Flatten(),
        Dense(16 * 8 * 8, 10, key=k[2], gain=gain),
        LinearLIF(tau=TAU, threshold=threshold),
    )


def frames(t=12, batch=3, size=32, channels=2, seed=1, scale=1.0):
    return jax.random.normal(jax.random.key(seed), (t, batch, size, size, channels)) * scale


# --- shapes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("padding", "stride", "expected"),
    [("SAME", 1, 32), ("SAME", 2, 16), ("VALID", 1, 30), ("VALID", 2, 15)],
)
def test_conv_output_spatial_size(padding, stride, expected):
    layer = Conv2d(2, 4, 3, key=jax.random.key(0), stride=stride, padding=padding)
    shape = (3, 32, 32, 2)
    assert layer.out_shape(shape) == (3, expected, expected, 4)
    _, y = layer(None, jnp.ones(shape))
    assert y.shape == layer.out_shape(shape), "declared out_shape must match reality"


def test_conv_rejects_wrong_channel_count():
    layer = Conv2d(2, 4, 3, key=jax.random.key(0))
    with pytest.raises(ValueError, match="expects 2 input channels"):
        layer.out_shape((3, 32, 32, 5))


def test_conv_rejects_bad_padding():
    with pytest.raises(ValueError, match="padding must be"):
        Conv2d(2, 4, 3, key=jax.random.key(0), padding="same")


def test_rectangular_kernels_and_strides():
    layer = Conv2d(1, 3, (3, 5), key=jax.random.key(0), stride=(1, 2), padding="VALID")
    shape = (2, 16, 20, 1)
    assert layer.out_shape(shape) == (2, 14, 8, 3)
    _, y = layer(None, jnp.ones(shape))
    assert y.shape == (2, 14, 8, 3)


def test_pool_and_flatten_shapes():
    pool = Pool2d(2)
    assert pool.out_shape((3, 32, 32, 8)) == (3, 16, 16, 8)
    flat = Flatten()
    assert flat.out_shape((3, 16, 16, 8)) == (3, 16 * 16 * 8)
    _, y = flat(None, jnp.ones((3, 16, 16, 8)))
    assert y.shape == (3, 2048)


def test_network_shape_propagates_end_to_end():
    assert convnet().out_shape((3, 32, 32, 2)) == (3, 10)


# --- numerics ----------------------------------------------------------------------------


def test_conv_matches_a_hand_computed_window():
    """A 1-channel 3x3 all-ones kernel over VALID padding is exactly a 3x3 box sum."""
    layer = Conv2d(1, 1, 3, key=jax.random.key(0), padding="VALID", use_bias=False)
    layer = eqx.tree_at(lambda m: m.weight, layer, jnp.ones((3, 3, 1, 1)))
    x = jnp.arange(25, dtype=jnp.float32).reshape(1, 5, 5, 1)
    _, y = layer(None, x)
    assert y.shape == (1, 3, 3, 1)
    assert y[0, 0, 0, 0] == pytest.approx(float(x[0, 0:3, 0:3, 0].sum()))
    assert y[0, 2, 2, 0] == pytest.approx(float(x[0, 2:5, 2:5, 0].sum()))


def test_avg_pool_averages_and_max_pool_maxes():
    x = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1)
    _, avg = Pool2d(2, mode="avg")(None, x)
    _, mx = Pool2d(2, mode="max")(None, x)
    assert avg[0, 0, 0, 0] == pytest.approx(float(x[0, 0:2, 0:2, 0].mean()))
    assert mx[0, 0, 0, 0] == pytest.approx(float(x[0, 0:2, 0:2, 0].max()))


def test_avg_pool_preserves_spike_count_where_max_pool_saturates():
    """Why avg is the SNN default: max discards how many units in the window fired."""
    one = jnp.zeros((1, 2, 2, 1)).at[0, 0, 0, 0].set(1.0)
    all_four = jnp.ones((1, 2, 2, 1))
    _, avg_one = Pool2d(2, mode="avg")(None, one)
    _, avg_all = Pool2d(2, mode="avg")(None, all_four)
    _, max_one = Pool2d(2, mode="max")(None, one)
    _, max_all = Pool2d(2, mode="max")(None, all_four)
    assert float(avg_one[0, 0, 0, 0]) < float(avg_all[0, 0, 0, 0])
    assert float(max_one[0, 0, 0, 0]) == float(max_all[0, 0, 0, 0]) == 1.0


def test_pool_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be"):
        Pool2d(2, mode="median")


def test_flatten_is_reversible_in_content():
    x = jax.random.normal(jax.random.key(0), (3, 4, 4, 2))
    _, y = Flatten()(None, x)
    assert jnp.array_equal(y.reshape(x.shape), x)


# --- integration -------------------------------------------------------------------------


def test_convnet_output_is_binary_and_spiking():
    net, xs = convnet(), frames()
    spikes, _ = unroll(net, xs)
    assert spikes.shape == (12, 3, 10)
    assert jnp.all((spikes == 0.0) | (spikes == 1.0))
    assert jnp.sum(spikes) > 0, "silent network makes the equivalence tests vacuous"


def test_convnet_parallel_matches_sequential():
    net, xs = convnet(), frames()
    seq, seq_state = unroll(net, xs)
    par, par_state = unroll_parallel(net, xs)
    assert jnp.array_equal(seq, par)
    assert jnp.allclose(seq_state[-1].v, par_state[-1].v, atol=1e-5)


def test_convnet_gradients_are_finite_and_reach_conv_weights():
    net, xs = convnet(), frames()

    def loss(m, x):
        return jnp.mean(unroll(m, x)[0])

    grads = eqx.filter_grad(loss)(net, xs)
    conv_grads = [grads.layers[0].weight, grads.layers[3].weight]
    for g in conv_grads:
        assert jnp.all(jnp.isfinite(g))
    assert any(jnp.any(g != 0) for g in conv_grads), "no gradient reached the conv layers"


def test_conv_layers_are_stateless():
    net = convnet()
    state = net.init_state((3, 32, 32, 2))
    assert state[0] is None and state[2] is None and state[6] is None
