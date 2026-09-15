"""The cell — one iteration of the local rule (spec §2.3).

```
m_uv  = MLP_msg([h_u, h_v, e_uv])          # gather over [V, K]
m_v   = concat(mean_K(m_uv), max_K(m_uv))  # double aggregation
q_vb  = MLP_bone([h_v, f_vb, w_vb])        # over [V, B]
q_v   = mean_B(q_vb)
h_v  <- GRUCell(h_v, [m_v, q_v, x_v])
dz_vb = MLP_out([h_v, f_vb, w_vb])
z_v  <- z_v + alpha * dz_v                 # damped delta, alpha = 0.25
w_v   = softmax(z_v / tau)
```

Two choices here are the design, and both are defended by ablation:

*Deltas, not absolute logits* (A4). Predicting `z` outright turns the system
into a feed-forward GNN wearing a costume — the last iteration overwrites
everything before it. Damped deltas make `dz -> 0` a natural fixed point, which
is the canonical form of a dynamical system and the precondition for P2.

*A GRU gate on the hidden state.* Bare NCAs diverge readily. The gate gives the
system a learned way to do nothing — the quiescence Reynolds' three rules have
no term for (spec §0.2).

Two deliberate details beyond the sketch above. The static per-vertex features
`x_v` join the GRU input: §2.2 computes them, and the gate is where static
context belongs. And `MLP_out`'s final layer is initialised at zero, so the
untrained cell starts as the identity — every trajectory begins at `dz = 0`,
which is both the stable starting point and the do-no-harm prior R3 wants.

Written against raw `mx` with fixed shapes and gather-only access, so
`mx.compile` sees real operations and nothing of ours sits in the hot loop.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import mlx.core as mx

from flock.data.preprocess.features import (
    EDGE_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    VERTEX_FEATURE_DIM,
)
from flock.domain.dynamics.state import CellConfig, CellState, StaticInputs

Params = dict[str, dict[str, mx.array]]


def _linear(key: mx.array, fan_in: int, fan_out: int, scale: float) -> dict[str, mx.array]:
    """Glorot-ish initialisation for one dense layer."""
    limit = scale * (6.0 / (fan_in + fan_out)) ** 0.5
    return {
        "w": mx.random.uniform(-limit, limit, (fan_in, fan_out), key=key),
        "b": mx.zeros((fan_out,)),
    }


def _mlp(
    keys: list[mx.array], fan_in: int, width: int, fan_out: int, final_scale: float = 1.0
) -> dict[str, mx.array]:
    """A two-layer MLP as a flat parameter dict."""
    first = _linear(keys[0], fan_in, width, 1.0)
    second = _linear(keys[1], width, fan_out, final_scale)
    return {"w1": first["w"], "b1": first["b"], "w2": second["w"], "b2": second["b"]}


def _apply_mlp(params: dict[str, mx.array], x: mx.array) -> mx.array:
    """Two-layer MLP with a GELU nonlinearity."""
    hidden = mx.matmul(x, params["w1"]) + params["b1"]
    return _mlp_tail(params, hidden)


def _mlp_tail(params: dict[str, mx.array], pre_activation: mx.array) -> mx.array:
    """GELU plus the second layer, given the first layer's pre-activation."""
    activated = pre_activation * mx.sigmoid(1.702 * pre_activation)
    return mx.matmul(activated, params["w2"]) + params["b2"]


def _project_broadcast(x: mx.array, weight: mx.array) -> mx.array:
    """Project `[N, V, C]` then broadcast over the neighbour/candidate axis.

    The first layer of an MLP over `concat([a, b, c])` is
    `a @ W_a + b @ W_b + c @ W_c`, so an operand that is constant along the
    gathered axis can be projected *before* it is broadcast rather than after.

    That matters here because `h_v` enters three MLPs broadcast over K=8 or B=8.
    Materialising `[N, V, 8, H]` and multiplying it by the full concatenated
    weight costs eight times the arithmetic and eight times the memory traffic
    of projecting `[N, V, H]` once. The result is identical; only the order of
    the sum changes.
    """
    return mx.expand_dims(mx.matmul(x, weight), 2)


def layer_params(params: Params, step: int, shared: bool) -> Params:
    """The parameter set in force at rollout step `step`.

    Shared weights are the hypothesis (P1): one rule, iterated. Unshared weights
    give every step its own parameters, which is baseline B3 and ablation A2 —
    the same depth at roughly T times the parameter count. Beyond the number of
    layers trained, indexing cycles, so a T-sweep runs without crashing; past
    that point an unshared model is repeating layers rather than extending, and
    only its native depth is a fair reading.
    """
    if shared:
        return params
    depth = next(iter(next(iter(params.values())).values())).shape[0]
    index = step % depth
    return {group: {key: value[index] for key, value in entries.items()}
            for group, entries in params.items()}


def init_params(config: CellConfig, seed: int, depth: int = 1) -> Params:
    """Initialise the parameter tree for one cell.

    With `config.shared_weights` the tree holds one set, reused at every step.
    Without it, every array grows a leading `depth` axis: same structure, same
    checkpoint format, `depth` times the parameters.
    """
    if not config.shared_weights:
        sets = [init_params(replace(config, shared_weights=True), seed * 977 + i)
                for i in range(depth)]
        return {
            group: {
                key: mx.stack([one[group][key] for one in sets])
                for key in sets[0][group]
            }
            for group in sets[0]
        }
    keys = list(mx.random.split(mx.random.key(seed), 10))
    hidden, width = config.hidden_dim, config.mlp_width

    message_in = 2 * hidden + EDGE_FEATURE_DIM
    bone_in = hidden + PAIR_FEATURE_DIM + 1
    gate_in = 3 * hidden + VERTEX_FEATURE_DIM  # [mean, max] messages, bone summary, static

    gru: dict[str, mx.array] = {}
    for index, gate in enumerate(("z", "r", "n")):
        layer = _linear(keys[4 + index], gate_in + hidden, hidden, 1.0)
        gru[f"w{gate}"] = layer["w"]
        gru[f"b{gate}"] = layer["b"]

    return {
        "msg": _mlp(keys[0:2], message_in, width, hidden),
        "bone": _mlp(keys[2:4], bone_in, width, hidden),
        "gru": gru,
        # Zero final layer: the untrained cell is the identity, so dz starts at 0.
        "out": _mlp(keys[7:9], bone_in, width, 1, final_scale=0.0),
    }


def gru_cell(params: dict[str, mx.array], hidden: mx.array, inputs: mx.array) -> mx.array:
    """Hand-written GRU update.

    Written out rather than pulled from a framework layer so that everything
    inside the compiled region is ours and visible (spec §3.5).
    """
    joined = mx.concatenate([inputs, hidden], axis=-1)
    update = mx.sigmoid(mx.matmul(joined, params["wz"]) + params["bz"])
    reset = mx.sigmoid(mx.matmul(joined, params["wr"]) + params["br"])
    candidate = mx.tanh(
        mx.matmul(mx.concatenate([inputs, reset * hidden], axis=-1), params["wn"]) + params["bn"]
    )
    return (1.0 - update) * candidate + update * hidden


def _gather_neighbours(hidden: mx.array, neighbours: mx.array) -> mx.array:
    """`[N, V, H]` gathered by `[N, V, K]` indices into `[N, V, K, H]`.

    Flattened to a single `take`: gather over a constant index table is the
    access pattern MLX/Metal handles best, and there is no scatter anywhere in
    the cell (spec §3.5).
    """
    batch, vertices, dim = hidden.shape
    offset = (mx.arange(batch, dtype=mx.int32) * vertices).reshape(batch, 1, 1)
    flat = (neighbours + offset).reshape(-1)
    return mx.take(hidden.reshape(batch * vertices, dim), flat, axis=0).reshape(
        batch, vertices, neighbours.shape[-1], dim
    )


def cell_step(
    params: Params,
    state: CellState,
    static: StaticInputs,
    config: CellConfig,
    step: int = 0,
) -> CellState:
    """Apply the cell once."""
    params = layer_params(params, step, config.shared_weights)
    hidden, logits = state.hidden, state.logits
    weights = mx.softmax(logits / config.temperature, axis=-1)

    # Mesh messages over the one-ring. The weight rows are split in the same
    # order the inputs were concatenated in: [h_u, h_v, e_uv].
    width = hidden.shape[-1]
    message_weight = params["msg"]["w1"]
    neighbour_hidden = _gather_neighbours(hidden, static.neighbours)
    message_pre = (
        mx.matmul(neighbour_hidden, message_weight[:width])
        + _project_broadcast(hidden, message_weight[width : 2 * width])
        + mx.matmul(static.edge_features, message_weight[2 * width :])
        + params["msg"]["b1"]
    )
    messages = _mlp_tail(params["msg"], message_pre)

    mask = mx.expand_dims(static.neighbour_mask, -1)
    live = mx.maximum(mx.sum(mask, axis=2), 1.0)
    # Named once and used by both reductions. Measured as no faster — the
    # compiler already eliminates the common subexpression — so this is for the
    # reader, not the clock.
    masked_messages = messages * mask
    mean_message = mx.sum(masked_messages, axis=2) / live
    # Masked slots must not win the max; -inf would poison the gradient.
    max_message = mx.max(masked_messages + (mask - 1.0) * 1e4, axis=2)

    # Bone channel: the long-range shortcut that makes a small T viable (H2).
    # Rows split as [h_v, f_vb, w_vb]; only h_v is constant along the candidate
    # axis, so only it is projected before broadcasting.
    pair_width = static.pair_features.shape[-1]
    expanded_weights = mx.expand_dims(weights, -1)
    bone_weight = params["bone"]["w1"]
    pair_contribution = (
        mx.matmul(static.pair_features, bone_weight[width : width + pair_width])
        + mx.matmul(expanded_weights, bone_weight[width + pair_width :])
        + params["bone"]["b1"]
    )
    bone_messages = _mlp_tail(
        params["bone"], _project_broadcast(hidden, bone_weight[:width]) + pair_contribution
    )
    candidate_mask = mx.expand_dims(static.candidate_mask, -1)
    bone_summary = mx.sum(bone_messages * candidate_mask, axis=2) / mx.maximum(
        mx.sum(candidate_mask, axis=2), 1.0
    )

    gate_input = mx.concatenate(
        [mean_message, max_message, bone_summary, static.vertex_features], axis=-1
    )
    new_hidden = gru_cell(params["gru"], hidden, gate_input)

    # Damped deltas on the logits: dz -> 0 is the fixed point (A4). Same input
    # layout as the bone MLP, but reading the *updated* hidden state.
    out_weight = params["out"]["w1"]
    delta_pre = (
        _project_broadcast(new_hidden, out_weight[:width])
        + mx.matmul(static.pair_features, out_weight[width : width + pair_width])
        + mx.matmul(expanded_weights, out_weight[width + pair_width :])
        + params["out"]["b1"]
    )
    delta = mx.squeeze(_mlp_tail(params["out"], delta_pre), -1)
    if config.absolute_logits:
        # Ablation A4: predict the logits outright instead of a damped delta.
        # This is the disguised feed-forward GNN — the last iteration overwrites
        # everything before it, and no fixed point is representable.
        new_logits = delta * static.candidate_mask
    else:
        new_logits = logits + config.delta_scale * delta * static.candidate_mask

    vertex_mask = mx.expand_dims(static.vertex_mask, -1)
    return CellState(
        hidden=new_hidden * vertex_mask,
        logits=new_logits * vertex_mask + logits * (1.0 - vertex_mask),
        age=state.age + 1,
    )


def weights_of(state: CellState, config: CellConfig, static: StaticInputs) -> mx.array:
    """Current weights, with masked candidate slots held at zero."""
    masked = state.logits + (static.candidate_mask - 1.0) * 1e4
    softmaxed: Any = mx.softmax(masked / config.temperature, axis=-1)
    weights: mx.array = softmaxed * static.candidate_mask
    return weights
