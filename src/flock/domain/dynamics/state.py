"""Cell state and configuration (spec §2.1, §2.3, §2.4).

The state is deliberately small: a hidden vector and a logit vector per vertex.
Everything else — graph, features, candidates — is static and precomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HIDDEN_DIM = 32
"""Default hidden width H (spec §3.4); swept by ablation A9 over {16, 32, 64}."""


@dataclass(frozen=True)
class CellConfig:
    """Hyperparameters of one cell (spec §2.3, §2.4, §3.4).

    The defaults give roughly 40-60k parameters — an order of magnitude under
    the brief's 1-2M ceiling, which is the point (spec H6).
    """

    hidden_dim: int = HIDDEN_DIM
    mlp_width: int = 96
    """96 by default; 128-192 is the fallback if the model underfits."""

    delta_scale: float = 0.25
    """α, the damping on logit updates. Damped deltas make Δz → 0 a natural
    fixed point; predicting logits outright would make this a disguised
    feed-forward GNN (spec §2.3)."""

    temperature: float = 1.0
    """Softmax temperature τ, fixed in V0."""

    fire_rate: float = 1.0
    """Per-vertex stochastic update probability. 1.0 disables it; ablation A7
    tests 0.5, the NCA setting known to harden attractors."""

    shared_weights: bool = True
    """False reproduces an unshared-depth baseline for ablation A2."""


@dataclass(frozen=True)
class CellState:
    """Mutable-per-iteration state of the dynamics, as backend arrays.

    Attributes:
        hidden: `[V, H]` hidden state, zero-initialised.
        logits: `[V, B]` candidate logits, initialised as `log(w_init + eps)`.
        age: Iterations this state has already lived through, used by the pool
            to sample supervision at a random T (spec §3.1).
    """

    hidden: Any
    logits: Any
    age: int = 0


@dataclass(frozen=True)
class StaticInputs:
    """Everything the cell reads but never writes, as backend arrays.

    Held separately from `CellState` because it is shared across a batch and
    must keep fixed shapes for `mx.compile` (spec §3.5).

    Attributes:
        vertex_features: `[V, Fv]` normal invariants, curvature, local area.
        edge_features: `[V, K, Fe]` per-neighbour geometric features.
        pair_features: `[V, B, Fp]` vertex-to-candidate-bone features.
        neighbours: `[V, K]` one-ring indices for the gather.
        neighbour_mask: `[V, K]` bool.
        candidate_mask: `[V, B]` bool.
        vertex_mask: `[V]` bool.
    """

    vertex_features: Any
    edge_features: Any
    pair_features: Any
    neighbours: Any
    neighbour_mask: Any
    candidate_mask: Any
    vertex_mask: Any

    def as_tree(self) -> dict[str, Any]:
        """Flatten to a dict of arrays.

        Compilers accept trees of arrays, not arbitrary dataclasses, so this is
        the form that crosses the `mx.compile` boundary. The dataclass is
        rebuilt on the inside, where it costs nothing and keeps the cell
        readable.
        """
        return {
            "vertex_features": self.vertex_features,
            "edge_features": self.edge_features,
            "pair_features": self.pair_features,
            "neighbours": self.neighbours,
            "neighbour_mask": self.neighbour_mask,
            "candidate_mask": self.candidate_mask,
            "vertex_mask": self.vertex_mask,
        }
