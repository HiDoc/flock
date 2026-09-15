"""Training meshes resident in device memory (spec §3.5).

Unified memory means the whole tier can sit as MLX arrays with no loader and no
host-device copy. At 1024 vertices a sample is a few hundred kilobytes, so even
several hundred meshes stay far under the budget.

This lives behind the backend port rather than in `training/` because residency
is exactly what differs between backends: a device with discrete memory would
stage and stream instead of holding everything, and the training loop should not
have to know which it is talking to.

Every array has a fixed shape. A single varying dimension would make
`mx.compile` recompile on each step, which is the failure R5 describes and the
first thing to suspect if throughput disappoints.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from flock.domain.sample import PreprocessedMesh


class MlxBank:
    """Every training mesh stacked into MLX arrays, gathered by index."""

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
        indices: NDArray[np.int64],
        pose_indices: NDArray[np.int64] | None = None,
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
