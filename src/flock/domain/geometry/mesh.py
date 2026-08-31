"""Mesh value object (spec §2.2, §5).

A mesh reaching this type has already been decimated, normalised and padded to
fixed shapes. Everything downstream may assume those properties hold, because
construction fails otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flock.domain.errors import InvariantError

# Fixed shapes (spec §2.2, §3.5): mx.compile recompiles on every new shape, so
# padding to constants is a hard requirement of the training loop, not a default.
NUM_VERTICES = 1024
NUM_NEIGHBOURS = 8
NUM_CANDIDATES = 8


@dataclass(frozen=True)
class Mesh:
    """A decimated, normalised, fixed-shape mesh.

    Attributes:
        positions: `[V, 3]` vertex positions, unit height and centred.
        normals: `[V, 3]` unit vertex normals.
        faces: `[F, 3]` triangle indices; padded rows are `-1`.
        neighbours: `[V, K]` one-ring indices, topped up by KNN when valence < K.
        neighbour_mask: `[V, K]` bool, False where the neighbour slot is padding.
        vertex_mask: `[V]` bool, False for padded vertices.
    """

    positions: NDArray[np.float32]
    normals: NDArray[np.float32]
    faces: NDArray[np.int32]
    neighbours: NDArray[np.int32]
    neighbour_mask: NDArray[np.bool_]
    vertex_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate every invariant this type promises."""
        v, k = NUM_VERTICES, NUM_NEIGHBOURS
        _expect_shape("positions", self.positions, (v, 3))
        _expect_shape("normals", self.normals, (v, 3))
        _expect_shape("neighbours", self.neighbours, (v, k))
        _expect_shape("neighbour_mask", self.neighbour_mask, (v, k))
        _expect_shape("vertex_mask", self.vertex_mask, (v,))
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise InvariantError(f"faces must be [F, 3], got {self.faces.shape}")

        if not self.vertex_mask.any():
            raise InvariantError("mesh has no real vertices")

        # Neighbour indices must address real vertices wherever the mask is set.
        live = self.neighbours[self.neighbour_mask]
        if live.size and (live.min() < 0 or live.max() >= v):
            raise InvariantError("neighbour indices out of range")
        if live.size and not self.vertex_mask[live].all():
            raise InvariantError("a live neighbour points at a padded vertex")

        # Faces must address real vertices; -1 marks padding rows.
        real_faces = self.faces[(self.faces >= 0).all(axis=1)]
        if real_faces.size and not self.vertex_mask[real_faces].all():
            raise InvariantError("a face references a padded vertex")

        norms = np.linalg.norm(self.normals[self.vertex_mask], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise InvariantError("normals are not unit length")

    @property
    def num_real_vertices(self) -> int:
        """Count of non-padded vertices."""
        return int(self.vertex_mask.sum())


def _expect_shape(name: str, array: NDArray[np.generic], shape: tuple[int, ...]) -> None:
    """Raise if `array` does not have exactly `shape`."""
    if array.shape != shape:
        raise InvariantError(f"{name} must be {shape}, got {array.shape}")
