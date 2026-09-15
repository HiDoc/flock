"""The `PreprocessedMesh` aggregate (spec §5).

One training sample, assembled once offline and thereafter read-only. This is
the aggregate root: it is the only object the data layer hands to training, and
its `validate()` runs on every load from disk.

That last point matters more than it looks. A stale or truncated `.npz` cache is
otherwise indistinguishable from a real one until it shows up as a
disappointing curve three days later — and the curve is what the gates read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flock.domain.errors import InvariantError
from flock.domain.geometry.mesh import Mesh
from flock.domain.geometry.pose import PoseBank
from flock.domain.geometry.skeleton import Skeleton
from flock.domain.skinning.corruption import CorruptionContext
from flock.domain.skinning.weights import CandidateTable, WeightField

SCHEMA_VERSION = 1
"""Bumped whenever the on-disk layout changes, so old caches fail loudly."""


@dataclass(frozen=True)
class PreprocessedMesh:
    """Everything one mesh contributes to training and evaluation.

    Attributes:
        mesh_id: Stable identifier, used for pool bookkeeping and splits.
        source: Dataset tier — `"d0"`, `"d1"` or `"d2"`.
        mesh: Decimated, normalised geometry.
        skeleton: The given bone hierarchy.
        candidates: Per-vertex candidate bones, from diffusion distance only.
        ground_truth: Reference weights, reprojected onto the decimated mesh.
        train_poses: Pose bank for `L_deform`.
        test_poses: Disjoint pose bank for evaluation.
        vertex_features: `[V, Fv]` precomputed per-vertex features.
        edge_features: `[V, K, Fe]` precomputed per-neighbour features.
        pair_features: `[V, B, Fp]` precomputed vertex-to-candidate features.
    """

    mesh_id: int
    source: str
    mesh: Mesh
    skeleton: Skeleton
    candidates: CandidateTable
    ground_truth: WeightField
    train_poses: PoseBank
    test_poses: PoseBank
    vertex_features: NDArray[np.float32]
    edge_features: NDArray[np.float32]
    pair_features: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate cross-object consistency.

        The individual value objects have already checked themselves; what is
        left is agreement *between* them.
        """
        self.validate()

    def corruption_context(self) -> CorruptionContext:
        """The narrow view of this mesh that corruption is allowed to see."""
        return CorruptionContext(
            neighbours=self.mesh.neighbours,
            neighbour_mask=self.mesh.neighbour_mask,
            vertex_mask=self.mesh.vertex_mask,
            candidate_bones=self.candidates.bones,
            candidate_mask=self.candidates.mask,
            candidate_distances=self.candidates.distances,
            bone_parents=self.skeleton.parents,
        )

    def validate(self) -> None:
        """Check every cross-cutting invariant of the aggregate."""
        j = self.skeleton.num_bones
        live_bones = self.candidates.bones[self.candidates.mask]
        if live_bones.size and live_bones.max() >= j:
            raise InvariantError(f"candidate bone index >= num_bones ({j})")

        for bank, label in ((self.train_poses, "train"), (self.test_poses, "test")):
            if bank.num_bones != j:
                raise InvariantError(
                    f"{label} pose bank has {bank.num_bones} bones, skeleton has {j}"
                )
            if bank.split != label:
                raise InvariantError(f"{label} pose bank is labelled {bank.split!r}")

        # Test poses must not appear in the training bank (spec §5 leak checklist).
        if _banks_overlap(self.train_poses, self.test_poses):
            raise InvariantError("train and test pose banks share a pose")

        v = self.mesh.positions.shape[0]
        if self.vertex_features.shape[0] != v:
            raise InvariantError("vertex_features has the wrong vertex count")
        if self.edge_features.shape[:2] != self.mesh.neighbours.shape:
            raise InvariantError("edge_features does not match the neighbour table")
        if self.pair_features.shape[:2] != self.candidates.bones.shape:
            raise InvariantError("pair_features does not match the candidate table")

        for name, array in (
            ("vertex_features", self.vertex_features),
            ("edge_features", self.edge_features),
            ("pair_features", self.pair_features),
        ):
            if not np.isfinite(array).all():
                raise InvariantError(f"{name} contains non-finite values")

        if self.source not in ("d0", "d1", "d2"):
            raise InvariantError(f"unknown source tier {self.source!r}")


def _banks_overlap(a: PoseBank, b: PoseBank) -> bool:
    """True if any pose transform appears in both banks."""
    flat_a = a.transforms.reshape(a.num_poses, -1)
    flat_b = b.transforms.reshape(b.num_poses, -1)
    if flat_a.shape[1] != flat_b.shape[1]:
        return False
    return any(np.isclose(flat_a, row, atol=1e-6).all(axis=1).any() for row in flat_b)
