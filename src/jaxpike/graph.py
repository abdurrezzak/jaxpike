"""Arbitrary network topologies: recurrence, skip connections, branching, fan-in.

`Sequential` is a straight chain. `Graph` lets you wire any layer to any other, including
back to itself, and does not ask whether the result is sensible.

    net = jp.Graph(
        nodes={
            "w_in":  jp.Dense(700, 128, key=k1),
            "hidden": jp.LIF(tau=20.0),
            "w_rec": jp.Dense(128, 128, key=k2),   # hidden feeding itself
            "w_out": jp.Dense(128, 20, key=k3),
            "out":   jp.LeakyIntegrator(tau=20.0),
        },
        edges=[
            ("input", "w_in"), ("w_in", "hidden"),
            ("hidden", "w_rec"), ("w_rec", "hidden"),   # the cycle
            ("hidden", "w_out"), ("w_out", "out"),
        ],
        output="out",
    )

Two rules make that well-defined, and they are the only two:

**A node with several incoming edges sums them**, which makes fan-in, skip connections and
residual paths work without special syntax. All incoming edges must agree in shape.

**An edge that closes a cycle reads the previous timestep.** A cycle cannot be resolved within
one step, so `Graph` finds the back-edges, evaluates everything else in topological order, and
feeds cycles from a one-step buffer -- exactly how an RNN is defined.

A `Graph` containing a cycle cannot run parallel-in-time and raises rather than silently
producing something else.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

INPUT = "input"


class GraphState(eqx.Module):
    """Per-node layer state plus the one-step buffer that feedback edges read from."""

    nodes: dict[str, Any]
    feedback: dict[str, Any]


def _validate(nodes: dict[str, Any], edges, output: str) -> None:
    if output not in nodes:
        raise ValueError(f"output {output!r} is not a node; have {sorted(nodes)}")
    if INPUT in nodes:
        raise ValueError(f"{INPUT!r} is reserved for the external input and cannot be a node")
    for src, dst in edges:
        if src != INPUT and src not in nodes:
            raise ValueError(f"edge ({src!r}, {dst!r}) has unknown source {src!r}")
        if dst not in nodes:
            raise ValueError(f"edge ({src!r}, {dst!r}) has unknown target {dst!r}")
    if not any(src == INPUT for src, _ in edges):
        raise ValueError(f"no edge from {INPUT!r}: the network is never fed anything")
    # A node with only outgoing edges would fail at the first timestep rather than here.
    fed = {dst for _, dst in edges}
    starved = sorted(set(nodes) - fed)
    if starved:
        raise ValueError(
            f"nodes {starved} have no incoming edge, so they have no input to compute from; "
            f"give each one an edge (from {INPUT!r} or from another node)"
        )


def _back_edges(nodes, edges) -> set[tuple[str, str]]:
    """Edges that close a cycle, found by depth-first search.

    These are the edges delayed by one timestep. Which edges get picked depends on traversal
    order, but any valid choice breaks the same cycles.
    """
    successors: dict[str, list[str]] = {name: [] for name in [INPUT, *nodes]}
    for src, dst in edges:
        successors[src].append(dst)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(successors, WHITE)
    found: set[tuple[str, str]] = set()

    def visit(node: str) -> None:
        color[node] = GRAY
        for nxt in successors[node]:
            if color[nxt] == GRAY:  # points at an ancestor: this closes a cycle
                found.add((node, nxt))
            elif color[nxt] == WHITE:
                visit(nxt)
        color[node] = BLACK

    for name in successors:
        if color[name] == WHITE:
            visit(name)
    return found


def _topological_order(nodes, edges, back: set[tuple[str, str]]) -> list[str]:
    forward = [(s, d) for s, d in edges if (s, d) not in back]

    # A subgraph connected only through its own cycle is orderable but can never receive a
    # value, so require reachability from the input along forward edges.
    reachable, frontier = {INPUT}, [INPUT]
    while frontier:
        current = frontier.pop()
        for src, dst in forward:
            if src == current and dst not in reachable:
                reachable.add(dst)
                frontier.append(dst)
    stranded = sorted(set(nodes) - reachable)
    if stranded:
        raise ValueError(
            f"nodes {stranded} are unreachable from {INPUT!r} along forward edges, so they "
            "can never receive a value; they are connected only through a cycle or not at all"
        )

    indegree = dict.fromkeys(nodes, 0)
    successors: dict[str, list[str]] = {name: [] for name in [INPUT, *nodes]}
    for src, dst in forward:
        successors[src].append(dst)
        indegree[dst] += 1

    queue = [INPUT] + [n for n, d in indegree.items() if d == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        if node != INPUT:
            order.append(node)
        for nxt in successors[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(nodes):
        unreached = sorted(set(nodes) - set(order))
        raise ValueError(
            f"nodes {unreached} are unreachable from {INPUT!r} or sit in a cycle that could "
            "not be broken; every node must be reachable by a forward path"
        )
    return order


class Graph(eqx.Module):
    """A network of layers wired by explicit edges."""

    nodes: dict[str, Any]
    edges: tuple[tuple[str, str], ...] = eqx.field(static=True)
    output: str = eqx.field(static=True)
    order: tuple[str, ...] = eqx.field(static=True)
    back: frozenset[tuple[str, str]] = eqx.field(static=True)

    def __init__(self, nodes: dict[str, Any], edges, output: str):
        edges = tuple(tuple(edge) for edge in edges)
        _validate(nodes, edges, output)
        back = _back_edges(nodes, edges)
        self.nodes = dict(nodes)
        self.edges = edges
        self.output = output
        self.back = frozenset(back)
        self.order = tuple(_topological_order(nodes, edges, back))

    @property
    def is_recurrent(self) -> bool:
        return bool(self.back)

    def incoming(self, name: str) -> list[tuple[str, bool]]:
        """Sources feeding `name`, each flagged with whether it is a delayed (feedback) edge."""
        return [(src, (src, dst) in self.back) for src, dst in self.edges if dst == name]

    def shapes(self, input_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Per-node output shape, resolved in topological order."""
        known: dict[str, tuple[int, ...]] = {INPUT: input_shape}
        for name in self.order:
            sources = self.incoming(name)
            resolved = [known[src] for src, _ in sources if src in known]
            if not resolved:
                raise ValueError(f"node {name!r} has no resolvable input shape")
            first = resolved[0]
            for shape in resolved[1:]:
                if shape != first:
                    raise ValueError(
                        f"node {name!r} sums its inputs, so they must share a shape; got "
                        f"{first} and {shape}"
                    )
            known[name] = self.nodes[name].out_shape(first)
        return known

    def init_state(self, input_shape: tuple[int, ...]) -> GraphState:
        shapes = self.shapes(input_shape)
        node_states = {
            name: self.nodes[name].init_state(
                shapes[self.incoming(name)[0][0]] if self.incoming(name) else input_shape
            )
            for name in self.order
        }
        # Only sources of feedback edges need a buffer, and it starts at zero.
        feedback = {src: jnp.zeros(shapes[src], dtype=jnp.float32) for src, _ in self.back}
        return GraphState(nodes=node_states, feedback=feedback)

    def out_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return self.shapes(input_shape)[self.output]

    def __call__(self, state: GraphState, x):
        values: dict[str, Any] = {INPUT: x}
        node_states = dict(state.nodes)

        for name in self.order:
            contributions = [
                state.feedback[src] if delayed else values[src]
                for src, delayed in self.incoming(name)
            ]
            total = contributions[0]
            for extra in contributions[1:]:
                total = total + extra
            node_states[name], values[name] = self.nodes[name](node_states[name], total)

        feedback = {src: values[src] for src, _ in self.back}
        return GraphState(nodes=node_states, feedback=feedback), values[self.output]

    def parallel_apply(self, state: GraphState, xs):
        """Only available for acyclic graphs -- a cycle in time cannot be unrolled in parallel."""
        from .parallel import supports_parallel

        if self.is_recurrent:
            raise TypeError(
                "this Graph is recurrent (feedback edges: "
                f"{sorted(self.back)}), so it cannot run parallel-in-time: timestep t "
                "genuinely depends on t-1. Use unroll() or unroll_checkpointed()."
            )
        values: dict[str, Any] = {INPUT: xs}
        node_states = dict(state.nodes)
        for name in self.order:
            layer = self.nodes[name]
            if not supports_parallel(layer):
                raise TypeError(
                    f"node {name!r} ({type(layer).__name__}) has no parallel_apply, so this "
                    "graph cannot run parallel-in-time."
                )
            contributions = [values[src] for src, _ in self.incoming(name)]
            total = contributions[0]
            for extra in contributions[1:]:
                total = total + extra
            node_states[name], values[name] = layer.parallel_apply(node_states[name], total)
        return GraphState(nodes=node_states, feedback={}), values[self.output]


__all__ = ["INPUT", "Graph", "GraphState"]
