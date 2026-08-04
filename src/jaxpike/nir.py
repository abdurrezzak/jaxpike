"""NIR (Neuromorphic Intermediate Representation) import and export.

NIR is the field's interchange format — ONNX for spiking networks. Exporting to it lets a
model trained here run in snnTorch, Norse, Spyx, Lava, Rockpool or Nengo, and deploy to Intel
Loihi, SpiNNaker2, BrainScaleS-2, SynSense Speck or Xylo. Requires ``pip install nir``.

**What NIR does and does not capture.** It standardizes *what a model computes* — the
primitives and their parameters — not how it is executed. So `unroll_parallel`, the fused
paths, and rematerialization have no representation here and need none: they are execution
strategies over a graph NIR already describes. What NIR cannot express is a *model* it has no
primitive for, and that is where exports fail rather than silently distort.

Two semantic gaps are worth knowing before relying on a round trip:

**Subtract reset has no NIR equivalent.** NIR's LIF resets the membrane to a fixed value
``v_reset``. Our default `reset="subtract"` removes exactly one threshold and keeps the
overshoot, which is a genuinely different model, so exporting it raises rather than quietly
substituting reset-to-zero and changing the model's behaviour.

**NIR specifies a differential equation, not a discretization.** Its LIF is
``tau*dv/dt = (v_leak - v) + R*I``; how an importing framework steps that is up to it. We use
the exact exponential solution, Norse uses forward Euler, and the two differ by O(dt/tau).
A NIR round trip therefore preserves the *model*, not bit-identical spike trains, once
another framework is involved.

**Units are not standardized by NIR, and this bites.** NIR stores `tau` in seconds; our
neurons store it in units of `dt`. Export therefore takes a `dt_seconds` argument declaring
what one of your timesteps physically means, and getting it wrong rescales every time
constant in the model. Verified interop notes as of nir 1.0.8: snnTorch reads our graphs, but
its importer assumes `dt = 1e-4 s` regardless of what the file says, and has no mapping for
the `LI` node at all — so a `LeakyIntegrator` readout will not cross into snnTorch. Round
trips within jaxpike are exact; anything leaving the library should be checked numerically on
the other side rather than assumed.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .conv import Conv2d, Flatten, Pool2d
from .layers import Dense, Sequential
from .neurons import LIF, LeakyIntegrator, LinearLIF


class NIRConversionError(Exception):
    """Raised when a model cannot be faithfully represented in NIR, or vice versa."""


def _require_nir():
    try:
        import nir
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("NIR support needs the `nir` package: pip install nir") from exc
    return nir


def _broadcast(value, shape) -> np.ndarray:
    """Per-unit parameter array shaped exactly like the layer's output.

    NIR infers a node's type from its parameter arrays, so a spatial neuron layer must carry
    (C, H, W)-shaped parameters -- a flat array would declare the layer as 1-D and fail type
    inference against the convolution feeding it.
    """
    target = tuple(int(d) for d in np.atleast_1d(shape))
    return np.broadcast_to(np.asarray(value, dtype=np.float32), target).copy()


def _nir_shape(shape: tuple[int, ...]) -> np.ndarray:
    """Our per-timestep shape as NIR declares it.

    NIR is channels-first throughout (PyTorch convention) while our conv layers are NHWC, so
    a spatial shape has to be reordered when it crosses the boundary.
    """
    feature = shape[1:]
    if len(feature) == 3:  # (H, W, C) -> (C, H, W)
        return np.array([feature[2], feature[0], feature[1]])
    return np.array(feature)


def _flatten_permutation(spatial: tuple[int, int, int]) -> np.ndarray:
    """Column permutation taking our HWC-flattened features to NIR's CHW-flattened order.

    This is easy to miss and silently wrong if skipped. Flattening (H, W, C) channels-last
    orders features as h-major then w then c; a channels-first framework flattening the same
    tensor orders them c-major then h then w. The two contain the same values in different
    positions, so the Dense layer that consumes them needs its columns permuted to match, or
    the exported model computes something subtly different from the original.
    """
    height, width, channels = spatial
    return np.transpose(np.arange(height * width * channels).reshape(spatial), (2, 0, 1)).reshape(
        -1
    )


# --- export --------------------------------------------------------------------------------


def _export_layer(
    layer, name: str, shape: tuple[int, ...], nir, dt_seconds: float
) -> tuple[list, tuple[int, ...]]:
    """Convert one layer to a list of (node_name, nir_node); returns them and the output shape."""
    units = _nir_shape(layer.out_shape(shape))

    if isinstance(layer, Dense):
        weight = np.asarray(layer.weight)
        if layer.bias is None:
            node = nir.Linear(weight=weight)
        else:
            node = nir.Affine(weight=weight, bias=np.asarray(layer.bias))
        return [(name, node)], layer.out_shape(shape)

    if isinstance(layer, Conv2d):
        # Ours is HWIO (kh, kw, in, out); NIR wants OIHW (out, in, kh, kw).
        weight = np.transpose(np.asarray(layer.weight), (3, 2, 0, 1))
        bias = (
            np.zeros(weight.shape[0], np.float32) if layer.bias is None else np.asarray(layer.bias)
        )
        node = nir.Conv2d(
            input_shape=(shape[-3], shape[-2]),
            weight=weight,
            stride=layer.stride,
            padding=layer.padding.lower(),
            dilation=(1, 1),
            groups=1,
            bias=bias,
        )
        return [(name, node)], layer.out_shape(shape)

    if isinstance(layer, Pool2d):
        if layer.mode != "avg":
            raise NIRConversionError(
                f"NIR has no max-pooling primitive, so Pool2d(mode='max') at '{name}' cannot "
                "be exported. Use mode='avg', which is the better default for binary spikes "
                "anyway because max saturates at 1 and discards how many units fired."
            )
        node = nir.AvgPool2d(kernel_size=layer.window, stride=layer.stride, padding=(0, 0))
        return [(name, node)], layer.out_shape(shape)

    if isinstance(layer, Flatten):
        # start_dim=0 because the declared type already excludes the batch axis.
        node = nir.Flatten(input_type={"input": _nir_shape(shape)}, start_dim=0, end_dim=-1)
        return [(name, node)], layer.out_shape(shape)

    if isinstance(layer, LIF):
        if layer.reset == "subtract":
            raise NIRConversionError(
                f"LIF at '{name}' uses reset='subtract', which NIR cannot express: its LIF "
                "resets the membrane to a fixed v_reset. Exporting it as reset-to-zero would "
                "silently change the model. Use reset='zero' if the model must be portable."
            )
        node = nir.LIF(
            tau=_broadcast(layer.tau * dt_seconds, units),
            r=_broadcast(1.0, units),
            v_leak=_broadcast(0.0, units),
            v_threshold=_broadcast(layer.threshold, units),
            v_reset=_broadcast(0.0, units),
        )
        return [(name, node)], layer.out_shape(shape)

    if isinstance(layer, LinearLIF):
        # Reset-free: a leaky integrator feeding a bare threshold. NIR has no single node for
        # this, but the two-node form is exact rather than an approximation.
        li = nir.LI(
            tau=_broadcast(jnp.exp(layer.log_tau) * dt_seconds, units),
            r=_broadcast(1.0, units),
            v_leak=_broadcast(0.0, units),
        )
        thr = nir.Threshold(threshold=_broadcast(layer.threshold, units))
        return [(f"{name}_li", li), (f"{name}_threshold", thr)], layer.out_shape(shape)

    if isinstance(layer, LeakyIntegrator):
        node = nir.LI(
            tau=_broadcast(jnp.exp(layer.log_tau) * dt_seconds, units),
            r=_broadcast(1.0, units),
            v_leak=_broadcast(0.0, units),
        )
        return [(name, node)], layer.out_shape(shape)

    raise NIRConversionError(
        f"{type(layer).__name__} at '{name}' has no NIR primitive. NIR covers LIF, leaky "
        "integrators, affine maps, convolution, average pooling and flatten; models outside "
        "that set (adaptive thresholds, Izhikevich dynamics, short-term plasticity) are not "
        "portable and must stay in jaxpike."
    )


def to_nir(module, input_shape: tuple[int, ...], *, dt_seconds: float = 1e-3):
    """Convert a `Sequential` (or single layer) to a `nir.NIRGraph`.

    `input_shape` is the per-timestep shape including the batch axis, e.g. `(1, 700)`.

    `dt_seconds` is **the physical duration of one of your timesteps**, and getting it wrong
    silently rescales every time constant in the exported model. Our neurons express `tau` in
    units of `dt`, so `tau=20` with the default `dt=1.0` means "20 timesteps" and says nothing
    about milliseconds. NIR, by contrast, stores `tau` in seconds. The default here assumes
    the usual SNN convention of a 1 ms step, making `tau=20` export as 0.02 s.
    """
    nir = _require_nir()
    layers = module.layers if isinstance(module, Sequential) else (module,)

    nodes: dict[str, Any] = {"input": nir.Input(input_type={"input": _nir_shape(input_shape)})}
    edges: list[tuple[str, str]] = []
    previous, shape = "input", input_shape
    pending_flatten: tuple[int, int, int] | None = None

    for index, layer in enumerate(layers):
        if isinstance(layer, Flatten) and len(shape) == 4:
            pending_flatten = (shape[1], shape[2], shape[3])
        elif isinstance(layer, Dense) and pending_flatten is not None:
            # Reorder columns from our HWC flatten order into NIR's CHW order.
            layer = _replace(
                layer,
                weight=jnp.asarray(
                    np.asarray(layer.weight)[:, _flatten_permutation(pending_flatten)]
                ),
            )
            pending_flatten = None

        produced, shape = _export_layer(layer, str(index), shape, nir, dt_seconds)
        for node_name, node in produced:
            nodes[node_name] = node
            edges.append((previous, node_name))
            previous = node_name

    nodes["output"] = nir.Output(output_type={"output": _nir_shape(shape)})
    edges.append((previous, "output"))
    return nir.NIRGraph(nodes=nodes, edges=edges)


def save(module, path, input_shape: tuple[int, ...], *, dt_seconds: float = 1e-3) -> None:
    """Write a model to a `.nir` file. See `to_nir` on why `dt_seconds` matters."""
    nir = _require_nir()
    nir.write(str(path), to_nir(module, input_shape, dt_seconds=dt_seconds))


# --- import --------------------------------------------------------------------------------


def _scalar(array, name: str, what: str) -> float:
    """NIR stores per-neuron parameters; our layers take one value unless it is uniform."""
    values = np.asarray(array).reshape(-1)
    if values.size and not np.allclose(values, values[0]):
        raise NIRConversionError(
            f"node '{name}' has per-neuron {what} and jaxpike layers currently take a single "
            f"value (range {values.min():.4g}-{values.max():.4g})."
        )
    return float(values[0]) if values.size else 0.0


def _check_standard_lif(node, name: str) -> None:
    if not np.allclose(np.asarray(node.v_leak), 0.0):
        raise NIRConversionError(
            f"node '{name}' has non-zero v_leak, which our neurons assume is 0."
        )
    if not np.allclose(np.asarray(node.r), 1.0):
        raise NIRConversionError(
            f"node '{name}' has resistance r != 1, which our neurons do not model. Fold it "
            "into the preceding weights before exporting."
        )


def _topological_order(graph) -> list[str]:
    """Node names from input to output. Assumes the chain topology `to_nir` produces."""
    successors: dict[str, list[str]] = {}
    indegree: dict[str, int] = dict.fromkeys(graph.nodes, 0)
    for src, dst in graph.edges:
        successors.setdefault(src, []).append(dst)
        indegree[dst] += 1

    queue = [n for n, d in indegree.items() if d == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nxt in successors.get(node, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(graph.nodes):
        raise NIRConversionError(
            "graph contains a cycle or disconnected nodes; only feedforward chains are "
            "supported for import."
        )
    return order


def from_nir(graph, *, dt: float = 1.0, dt_seconds: float = 1e-3):
    """Convert a `nir.NIRGraph` back into a `Sequential`.

    Handles the feedforward chains `to_nir` produces, including recognising an `LI` immediately
    followed by a `Threshold` as a `LinearLIF`.

    `dt_seconds` must match what the file was written with, since NIR stores `tau` in seconds
    and our neurons want it in timesteps.
    """
    nir = _require_nir()
    order = _topological_order(graph)
    layers: list[Any] = []
    skip = set()
    pending_flatten: tuple[int, int, int] | None = None

    for position, name in enumerate(order):
        if name in skip:
            continue
        node = graph.nodes[name]

        if isinstance(node, nir.Input | nir.Output):
            continue

        if isinstance(node, nir.Affine | nir.Linear):
            raw = np.asarray(node.weight)
            if pending_flatten is not None:
                # Invert the export-side reordering: put CHW-ordered columns back into HWC.
                inverse = np.argsort(_flatten_permutation(pending_flatten))
                raw = raw[:, inverse]
                pending_flatten = None
            weight = jnp.asarray(raw)
            bias = getattr(node, "bias", None)
            layer = Dense(weight.shape[1], weight.shape[0], key=jax.random.key(0))
            layer = _replace(layer, weight=weight)
            layer = _replace(layer, bias=None if bias is None else jnp.asarray(np.asarray(bias)))
            layers.append(layer)

        elif isinstance(node, nir.Conv2d):
            weight = np.transpose(np.asarray(node.weight), (2, 3, 1, 0))  # OIHW -> HWIO
            layer = Conv2d(
                weight.shape[2],
                weight.shape[3],
                (weight.shape[0], weight.shape[1]),
                key=jax.random.key(0),
                stride=tuple(np.atleast_1d(node.stride).tolist() * 2)[:2],
                padding="SAME" if str(node.padding).lower() == "same" else "VALID",
            )
            layer = _replace(layer, weight=jnp.asarray(weight))
            if node.bias is not None:
                layer = _replace(layer, bias=jnp.asarray(np.asarray(node.bias)))
            layers.append(layer)

        elif isinstance(node, nir.AvgPool2d):
            window = tuple(np.atleast_1d(node.kernel_size).tolist() * 2)[:2]
            stride = tuple(np.atleast_1d(node.stride).tolist() * 2)[:2]
            layers.append(Pool2d(window, stride=stride, mode="avg"))

        elif isinstance(node, nir.Flatten):
            declared = np.asarray(node.input_type["input"]).reshape(-1)
            if declared.size == 3:  # (C, H, W) -> remember as our (H, W, C)
                pending_flatten = (int(declared[1]), int(declared[2]), int(declared[0]))
            layers.append(Flatten())

        elif isinstance(node, nir.LIF):
            _check_standard_lif(node, name)
            if not np.allclose(np.asarray(node.v_reset), 0.0):
                raise NIRConversionError(
                    f"node '{name}' resets to a non-zero v_reset, which our LIF does not model."
                )
            layers.append(
                LIF(
                    tau=_scalar(node.tau, name, "tau") / dt_seconds,
                    threshold=_scalar(node.v_threshold, name, "threshold"),
                    dt=dt,
                    reset="zero",
                )
            )

        elif isinstance(node, nir.LI):
            _check_standard_lif(node, name)
            tau = _scalar(node.tau, name, "tau") / dt_seconds
            # LI followed immediately by Threshold is a reset-free spiking neuron.
            nxt = order[position + 1] if position + 1 < len(order) else None
            if nxt is not None and isinstance(graph.nodes[nxt], nir.Threshold):
                skip.add(nxt)
                threshold = _scalar(graph.nodes[nxt].threshold, nxt, "threshold")
                layers.append(LinearLIF(tau=tau, threshold=threshold, dt=dt))
            else:
                layers.append(LeakyIntegrator(tau=tau, dt=dt))

        elif isinstance(node, nir.Threshold):
            raise NIRConversionError(
                f"node '{name}' is a bare Threshold not preceded by an LI; jaxpike has no "
                "stateless thresholding layer."
            )

        else:
            raise NIRConversionError(
                f"node '{name}' is a {type(node).__name__}, which has no jaxpike equivalent."
            )

    return Sequential(*layers)


def load(path, *, dt: float = 1.0, dt_seconds: float = 1e-3):
    """Read a `.nir` file into a `Sequential`. `dt_seconds` must match the writer's."""
    nir = _require_nir()
    return from_nir(nir.read(str(path)), dt=dt, dt_seconds=dt_seconds)


def _replace(module, **changes):
    import equinox as eqx

    for field, value in changes.items():
        module = eqx.tree_at(
            lambda m, f=field: getattr(m, f), module, value, is_leaf=lambda x: x is None
        )
    return module


__all__ = ["NIRConversionError", "from_nir", "load", "save", "to_nir"]
