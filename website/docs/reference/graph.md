---
id: graph
title: Graph
sidebar_position: 4
---

# Graph

`Graph` wires layers into an arbitrary directed graph: recurrence, skip connections,
branching and fan-in. Where [`Sequential`](./layers.md#sequential) applies layers in order,
`Graph` takes a set of named nodes and the edges between them.

```python
jp.Graph(nodes: dict[str, Module], edges: list[tuple[str, str]], output: str)
```

| Argument | Meaning |
|---|---|
| `nodes` | Named layers. Any module following the [state contract](./neurons.md#the-state-contract). |
| `edges` | `(source, destination)` pairs. The literal name `"input"` denotes the network input. |
| `output` | Name of the node whose output the graph returns. |

## Two rules

Everything `Graph` does follows from two rules, and there are no others.

**A node with several incoming edges sums them.** This is what a synapse does, and it makes
fan-in and skip connections work without special syntax.

**An edge that closes a cycle reads the previous timestep.** A cycle cannot be resolved within
a single step, so back-edges carry state forward in time — which is exactly what makes a
recurrent spiking network recurrent. `Graph` identifies back-edges automatically by finding a
topological order over the remaining edges.

## Recurrent network

```python
import jax
import jaxpike as jp

k1, k2, k3 = jax.random.split(jax.random.key(0), 3)

net = jp.Graph(
    nodes={
        "w_in": jp.Dense(700, 128, key=k1),
        "hidden": jp.LIF(tau=20.0),
        "w_rec": jp.Dense(128, 128, key=k2),
        "w_out": jp.Dense(128, 20, key=k3),
        "out": jp.LeakyIntegrator(tau=20.0),
    },
    edges=[
        ("input", "w_in"),
        ("w_in", "hidden"),
        ("hidden", "w_rec"),
        ("w_rec", "hidden"),      # closes a cycle
        ("hidden", "w_out"),
        ("w_out", "out"),
    ],
    output="out",
)

xs = jax.random.uniform(jax.random.key(1), (250, 32, 700))
membrane, state = jp.unroll(net, xs)
```

## Skip connections

An edge that does not close a cycle is resolved within the timestep, so a skip connection is
just an extra edge. The destination sums its inputs:

```python
net = jp.Graph(
    nodes={
        "w1": jp.Dense(64, 64, key=k1),
        "lif1": jp.LIF(tau=20.0),
        "w2": jp.Dense(64, 64, key=k2),
        "lif2": jp.LIF(tau=20.0),
    },
    edges=[
        ("input", "w1"),
        ("w1", "lif1"),
        ("lif1", "w2"),
        ("w2", "lif2"),
        ("lif1", "lif2"),   # skip: summed with w2's output
    ],
    output="lif2",
)
```

## State

`Graph` state is a `GraphState` holding each node's state plus the values carried across
back-edges. As with every other module, it is explicit and functional, so a long sequence can
be processed in chunks:

```python
out_a, state = jp.unroll(net, xs[:100])
out_b, state = jp.unroll(net, xs[100:], state)
```

## Limitations

**A recurrent graph cannot run parallel-in-time.** Recurrence is a genuine cycle in time, so
`unroll_parallel` raises rather than quietly computing something else. Use `unroll` or
`unroll_checkpointed`.

**Every node must be reachable from `"input"`** and must reach `output`; a graph with
unreachable nodes is rejected at construction rather than silently ignoring them.

## See also

- [Topologies guide](../guides/topologies.md) — worked examples and design discussion
- [Execution](./execution.md) — which execution strategies apply to graphs
