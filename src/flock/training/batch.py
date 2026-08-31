"""Batching preprocessed meshes into backend arrays (spec §3.5).

The whole dataset is resident in unified memory as MLX arrays: no loader, no
host-to-device copy, no pipeline. At 1024 vertices a sample is a few hundred
kilobytes, so even the full D1 tier is well under the memory budget.

Every array here has a fixed shape. A single varying dimension anywhere would
make `mx.compile` recompile on each step, which is the failure R5 describes and
the first thing to suspect if throughput disappoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from flock.domain.dynamics.state import StaticInputs
from flock.domain.sample import PreprocessedMesh

LOGIT_EPSILON = 1e-6
"""Floor inside `log(w + eps)`, so a zero weight maps to a finite logit."""


@dataclass(frozen=True)
class Batch:
    """One training batch, entirely on device."""

    static: StaticInputs
    target: mx.array
    """`[N, V, B]` ground-truth weights over candidate slots."""

    initial_logits: mx.array
    """`[N, V, B]` starting logits, corrupted or clean."""

    positions: mx.array
    bones: mx.array
    train_poses: mx.array


def logits_from_weights(weights: np.ndarray) -> np.ndarray:
    """`z = log(w + eps)`, the initialisation of spec §2.1."""
    logits: np.ndarray = np.log(np.maximum(weights, 0.0) + LOGIT_EPSILON).astype(np.float32)
    return logits


def build_batch(
    samples: list[PreprocessedMesh],
    initial_weights: np.ndarray,
) -> Batch:
    """Stack samples into device arrays.

    Args:
        samples: The meshes in this batch.
        initial_weights: `[N, V, B]` starting weights, already corrupted.
    """
    stack = np.stack
    static = StaticInputs(
        vertex_features=mx.array(stack([s.vertex_features for s in samples])),
        edge_features=mx.array(stack([s.edge_features for s in samples])),
        pair_features=mx.array(stack([s.pair_features for s in samples])),
        neighbours=mx.array(stack([s.mesh.neighbours for s in samples]).astype(np.int32)),
        neighbour_mask=mx.array(
            stack([s.mesh.neighbour_mask for s in samples]).astype(np.float32)
        ),
        candidate_mask=mx.array(stack([s.candidates.mask for s in samples]).astype(np.float32)),
        vertex_mask=mx.array(stack([s.mesh.vertex_mask for s in samples]).astype(np.float32)),
    )
    return Batch(
        static=static,
        target=mx.array(stack([s.ground_truth.values for s in samples])),
        initial_logits=mx.array(logits_from_weights(initial_weights)),
        positions=mx.array(stack([s.mesh.positions for s in samples])),
        bones=mx.array(stack([s.candidates.bones for s in samples]).astype(np.int32)),
        train_poses=mx.array(stack([s.train_poses.transforms for s in samples])),
    )


class MeshBank:
    """Every training mesh resident on device, gathered by index.

    Unified memory means the whole tier can sit as MLX arrays with no loader and
    no host-device copy (spec §3.5). At 1024 vertices a few hundred meshes cost
    well under a gigabyte, and §5 puts V0's requirement at 100-500 anyway.
    """

    def __init__(self, samples: list[PreprocessedMesh]) -> None:
        """Stack the samples into device arrays."""
        self.samples = samples
        stack = np.stack
        self.vertex_features = mx.array(stack([s.vertex_features for s in samples]))
        self.edge_features = mx.array(stack([s.edge_features for s in samples]))
        self.pair_features = mx.array(stack([s.pair_features for s in samples]))
        self.neighbours = mx.array(stack([s.mesh.neighbours for s in samples]).astype(np.int32))
        self.neighbour_mask = mx.array(
            stack([s.mesh.neighbour_mask for s in samples]).astype(np.float32)
        )
        self.candidate_mask = mx.array(
            stack([s.candidates.mask for s in samples]).astype(np.float32)
        )
        self.vertex_mask = mx.array(stack([s.mesh.vertex_mask for s in samples]).astype(np.float32))
        self.target = mx.array(stack([s.ground_truth.values for s in samples]))
        self.positions = mx.array(stack([s.mesh.positions for s in samples]))
        self.bones = mx.array(stack([s.candidates.bones for s in samples]).astype(np.int32))
        self.train_poses = mx.array(stack([s.train_poses.transforms for s in samples]))

    def __len__(self) -> int:
        """Number of meshes held."""
        return len(self.samples)

    def gather(
        self,
        indices: np.ndarray,
        pose_indices: np.ndarray | None = None,
    ) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        """Static inputs and targets for the given mesh indices.

        Args:
            indices: Which meshes to gather.
            pose_indices: Which poses to include. Spec §3.3 draws 2-4 per step
                from the bank of 16-32; posing the whole bank instead makes
                `L_deform` dominate the step cost for no extra signal.

        Returns:
            `(static_tree, extras)` — the tree crosses the compile boundary,
            extras carry the targets and geometry the losses need.
        """
        picks = mx.array(indices.astype(np.int32))
        static = {
            "vertex_features": mx.take(self.vertex_features, picks, axis=0),
            "edge_features": mx.take(self.edge_features, picks, axis=0),
            "pair_features": mx.take(self.pair_features, picks, axis=0),
            "neighbours": mx.take(self.neighbours, picks, axis=0),
            "neighbour_mask": mx.take(self.neighbour_mask, picks, axis=0),
            "candidate_mask": mx.take(self.candidate_mask, picks, axis=0),
            "vertex_mask": mx.take(self.vertex_mask, picks, axis=0),
        }
        poses = mx.take(self.train_poses, picks, axis=0)
        if pose_indices is not None:
            poses = mx.take(poses, mx.array(pose_indices.astype(np.int32)), axis=1)
        extras = {
            "target": mx.take(self.target, picks, axis=0),
            "positions": mx.take(self.positions, picks, axis=0),
            "bones": mx.take(self.bones, picks, axis=0),
            "poses": poses,
        }
        return static, extras
