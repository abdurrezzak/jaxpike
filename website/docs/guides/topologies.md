---
id: topologies
title: Arbitrary topologies with Graph
sidebar_position: 4
---

# Arbitrary topologies with `Graph`

`Sequential` is a straight chain. `Graph` wires any layer to any other — recurrence, skip
connections, branching, fan-in — and does not ask whether the result is sensible.

```python
net = jp.Graph(
    nodes={
        "w_in": jp.Dense(700, 128, key=k1),
        "hidden": jp.LIF(tau=20.0),
        "w_rec": jp.Dense(128, 128, key=k2),   # hidden feeding itself
        "w_out": jp.Dense(128, 20, key=k3),
        "out": jp.LeakyIntegrator(tau=20.0),
    },
    edges=[
        ("input", "w_in"), ("w_in", "hidden"),
        ("hidden", "w_rec"), ("w_rec", "hidden"),   # the cycle
        ("hidden", "w_out"), ("w_out", "out"),
    ],
    output="out",
)

spikes, state = jp.unroll(net, xs)
```

`"input"` is reserved for the external input and cannot be a node name. `output` names the node
whose value `unroll` returns.

<img src="/img/figures/architecture_light.png" alt="Architecture diagrams" className="figure-light" />
<img src="/img/figures/architecture_dark.png" alt="Architecture diagrams" className="figure-dark" />

## The two rules

Everything about `Graph` follows from these, and there are no others.

**A node with several incoming edges sums them.** That is what a synapse does, and it makes
fan-in, skip connections and residual paths work with no special syntax. All incoming edges
must agree in shape.

**An edge that closes a cycle reads the previous timestep.** A cycle cannot be resolved within
one step without infinite regress, so `Graph` finds the back-edges, evaluates everything else
in topological order, and feeds cycles from a one-step buffer. This is exactly how an RNN is
defined, and it is what makes a recurrent SNN recurrent.

You do not declare back-edges. `Graph` finds them at construction and exposes them:

```python
net.is_recurrent      # True
net.back              # frozenset of the edges that were made delayed
net.incoming("hidden")  # [("w_in", False), ("w_rec", True)] -- True means delayed
net.order             # the evaluation order
```

## Skip connections

No cycle, so no delay — a skip is just two edges arriving at the same node and summing:

```python
net = jp.Graph(
    nodes={
        "w1": jp.Dense(128, 128, key=k1),
        "n1": jp.LIF(tau=20.0),
        "w2": jp.Dense(128, 128, key=k2),
        "n2": jp.LIF(tau=20.0),
    },
    edges=[
        ("input", "w1"), ("w1", "n1"),
        ("n1", "w2"), ("w2", "n2"),
        ("n1", "n2"),          # skip: summed into n2 with w2's output
    ],
    output="n2",
)
```

## Recurrence needs a much smaller gain

The recurrent weight's output is summed into the same membrane on the next timestep, so a gain
sized for feedforward drive makes the loop self-amplifying and the network saturates within a
few timesteps. The SHD recurrent model uses `gain=0.2` on the recurrent weight against
`jp.lif_gain(20.0)` — about 6.3 — on the feedforward ones. See
[Why deep SNNs go silent](./silent-networks.md).

## What recurrence costs

**Parallel-in-time no longer applies.** A genuine cycle in time cannot be solved by an
associative scan, so a recurrent `Graph` refuses `unroll_parallel` by name, listing the
offending node. A silent fallback would turn a correctness problem into a performance mystery.

**e-prop refuses recurrent graphs too.** The eligibility-trace factorization drops recurrent
paths, so accepting one would return a quietly wrong gradient. It raises instead. See
[Online learning](./online-learning.md).

Both restrictions are checked by name with the offending node reported, not by a silent
capability test.

## State

`GraphState` holds per-node layer state plus the one-step buffer that feedback edges read from.
It threads through `unroll` like any other state, so chunking a long sequence stays exact:

```python
spikes_a, state = jp.unroll(net, xs[:50])
spikes_b, state = jp.unroll(net, xs[50:], state)
```

## Drawing it

```python
from jaxpike import viz

viz.architecture(net, input_shape=(1, 700))
```

Nodes are laid out in evaluation order with shapes annotated and feedback edges drawn
distinctly, which is the fastest way to confirm that the wiring you wrote is the wiring you
meant.
