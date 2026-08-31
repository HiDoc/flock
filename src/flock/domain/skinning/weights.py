"""Weight field and candidate table (spec §2.1, §2.2, §4.4).

`CandidateTable` is the type that encodes risk R7. Candidate selection may read
geometry only; it never sees ground-truth weights. That is enforced here by the
constructor signature and by `from_diffusion_distance`, which is the only
supported way to build one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flock.domain.errors import InvariantError
from flock.domain.geometry.mesh import NUM_CANDIDATES, NUM_VERTICES

WEIGHT_SUM_TOL = 1e-4
SUPPORT_THRESHOLD = 1e-3
"""Support cutoff, reported alongside every support metric (spec H5, §4.1)."""


@dataclass(frozen=True)
class CandidateTable:
    """The B bones each vertex is allowed to be influenced by.

    Built from diffusion distance so that geodesically distant bones are excluded
    before the dynamics ever runs — the injected geodesy of spec H1.

    Attributes:
        bones: `[V, B]` bone indices into the skeleton.
        mask: `[V, B]` bool, False where the mesh has fewer than B bones to offer.
        distances: `[V, B]` diffusion distance to each candidate bone.
    """

    bones: NDArray[np.int32]
    mask: NDArray[np.bool_]
    distances: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate shapes, index range and per-row uniqueness."""
        shape = (NUM_VERTICES, NUM_CANDIDATES)
        for name, array in (
            ("bones", self.bones),
            ("mask", self.mask),
            ("distances", self.distances),
        ):
            if array.shape != shape:
                raise InvariantError(f"{name} must be {shape}, got {array.shape}")

        live = self.bones[self.mask]
        if live.size and live.min() < 0:
            raise InvariantError("candidate bone index is negative")
        if (self.distances[self.mask] < 0).any():
            raise InvariantError("candidate distance is negative")

        # A bone appearing twice in one row silently halves the effective budget.
        for row, row_mask in zip(self.bones, self.mask, strict=True):
            selected = row[row_mask]
            if np.unique(selected).size != selected.size:
                raise InvariantError("a vertex lists the same candidate bone twice")

    @classmethod
    def from_diffusion_distance(
        cls,
        distances: NDArray[np.float32],
        vertex_mask: NDArray[np.bool_],
    ) -> CandidateTable:
        """Select the top-B nearest bones per vertex by diffusion distance.

        This signature accepts no weights of any kind. Selecting candidates with
        knowledge of the ground truth would put an unreachable ceiling under
        every later metric (spec R1) while making the results irreproducible
        (spec R7).

        Args:
            distances: `[V, J]` vertex-to-bone diffusion distances.
            vertex_mask: `[V]` bool marking real vertices.

        Returns:
            A validated candidate table.
        """
        num_vertices, num_bones = distances.shape
        take = min(NUM_CANDIDATES, num_bones)
        order = np.argsort(distances, axis=1, kind="stable")[:, :take]

        bones = np.zeros((num_vertices, NUM_CANDIDATES), dtype=np.int32)
        mask = np.zeros((num_vertices, NUM_CANDIDATES), dtype=bool)
        chosen = np.zeros((num_vertices, NUM_CANDIDATES), dtype=np.float32)
        bones[:, :take] = order.astype(np.int32)
        chosen[:, :take] = np.take_along_axis(distances, order, axis=1)
        mask[:, :take] = True
        # Padded vertices hold no candidates; a masked slot must never be read.
        mask &= vertex_mask[:, None]
        return cls(bones, mask, chosen)


@dataclass(frozen=True)
class WeightField:
    """Skinning weights over a candidate table.

    Weights are stored per candidate slot, not per skeleton bone: column `b` of
    row `v` is the weight of `candidates.bones[v, b]`.

    Attributes:
        values: `[V, B]` non-negative weights summing to 1 per real vertex.
    """

    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        """Validate shape, sign and normalisation."""
        shape = (NUM_VERTICES, NUM_CANDIDATES)
        if self.values.shape != shape:
            raise InvariantError(f"values must be {shape}, got {self.values.shape}")
        if not np.isfinite(self.values).all():
            raise InvariantError("weights contain non-finite values")
        if (self.values < 0).any():
            raise InvariantError("weights contain negative values")

        sums = self.values.sum(axis=1)
        # Padded rows are all-zero; real rows must be a partition of unity.
        offenders = ~(np.isclose(sums, 1.0, atol=WEIGHT_SUM_TOL) | np.isclose(sums, 0.0))
        if offenders.any():
            worst = float(sums[offenders].max())
            raise InvariantError(f"weight rows must sum to 1 or 0; found a row summing to {worst}")

    def support(self, threshold: float = SUPPORT_THRESHOLD) -> NDArray[np.bool_]:
        """Boolean support mask at an explicit threshold.

        Softmax never emits exact zeros, so the threshold is a reported
        parameter of the metric rather than a hidden constant (spec H5).
        """
        return self.values > threshold
