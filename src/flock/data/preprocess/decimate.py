"""Welding and decimation to the fixed vertex budget (spec §5, ADR-0003).

Corpus meshes run to a median of ~4k vertices and a maximum of 199k, against a
fixed V=1024. Decimation is mandatory for most of the tier.

Two properties of this corpus dictate the method.

**Meshes arrive as split-seam soup.** Between 3% and 51% of vertices are exact
positional duplicates — the usual result of exporting per-face normals and UVs
through FBX — which shatters a character into hundreds or thousands of nominal
components. One mesh measured 2,835 components before welding and 21 after.
Since the whole point of injecting geodesy (H1) is to respect connectivity,
welding coincident vertices is a correctness step, not a tidy-up.

**QEM is not usable here.** `igl.decimate` assumes manifold input; on these
meshes it silently returned 8,506 vertices for a 2,838-face budget on one model
and *zero* vertices on two others. So decimation is by vertex clustering
instead: quantise to a grid, keep one representative per occupied cell. It
cannot fail, it hits the budget by construction, and it yields a birth map that
makes weight transfer an exact gather.
"""

from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np
from numpy.typing import NDArray

WELD_TOLERANCE = 1e-6
"""Relative to mesh extent: merges exported split seams, not distinct geometry."""

MAX_SEARCH_STEPS = 24


@dataclass(frozen=True)
class Decimated:
    """A decimated mesh plus the birth map back to its source.

    Attributes:
        positions: `[n, 3]` surviving vertex positions, `n <= budget`.
        faces: `[f, 3]` non-degenerate triangles indexing `positions`.
        source_vertex: `[n]` index of the source vertex each one came from.
    """

    positions: NDArray[np.float64]
    faces: NDArray[np.int32]
    source_vertex: NDArray[np.int64]


def weld(
    positions: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int64]]:
    """Merge coincident vertices and reindex the faces.

    Returns:
        `(positions, faces, source_vertex)` where `source_vertex` maps each
        welded vertex back to one of the source vertices it absorbed.
    """
    extent = float(np.abs(positions - positions.mean(axis=0)).max())
    tolerance = WELD_TOLERANCE * max(extent, 1e-12)
    welded, source_vertex, _, welded_faces = igl.remove_duplicate_vertices(
        positions, faces, tolerance
    )
    return welded, welded_faces.astype(np.int32), source_vertex.astype(np.int64)


def _cluster(
    positions: NDArray[np.float64],
    faces: NDArray[np.int32],
    resolution: int,
) -> Decimated:
    """Collapse vertices onto a `resolution^3` grid, one representative per cell."""
    low, high = positions.min(axis=0), positions.max(axis=0)
    span = np.maximum(high - low, 1e-12)
    cell = np.clip(((positions - low) / span * resolution).astype(np.int64), 0, resolution - 1)
    key = (cell[:, 0] * resolution + cell[:, 1]) * resolution + cell[:, 2]

    _, representative, inverse = np.unique(key, return_index=True, return_inverse=True)
    kept_positions = positions[representative]

    remapped = inverse[faces]
    non_degenerate = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    kept_faces = remapped[non_degenerate]

    # Drop vertices no surviving face references. Collapsing a cell can orphan
    # its representative, and geometry-central segfaults on an unreferenced
    # vertex rather than reporting one.
    used = np.unique(kept_faces) if len(kept_faces) else np.zeros(0, dtype=np.int64)
    relabel = np.full(len(kept_positions), -1, dtype=np.int64)
    relabel[used] = np.arange(len(used))
    return Decimated(
        kept_positions[used],
        relabel[kept_faces].astype(np.int32) if len(kept_faces) else kept_faces.astype(np.int32),
        representative[used].astype(np.int64),
    )


def decimate_to_vertex_budget(
    positions: NDArray[np.float64],
    faces: NDArray[np.int32],
    budget: int,
) -> Decimated:
    """Weld, then cluster down to at most `budget` vertices.

    The grid resolution is binary-searched, keeping the finest grid that still
    fits. Overshooting is unrecoverable — padding is free, but exceeding V=1024
    would break the fixed shapes the compiler depends on.
    """
    welded_positions, welded_faces, birth = weld(positions, faces)
    if len(welded_positions) <= budget:
        return Decimated(welded_positions, welded_faces, birth)

    best: Decimated | None = None
    low, high = 2, 512
    for _ in range(MAX_SEARCH_STEPS):
        if low > high:
            break
        mid = (low + high) // 2
        candidate = _cluster(welded_positions, welded_faces, mid)
        if len(candidate.positions) <= budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1

    if best is None:
        best = _cluster(welded_positions, welded_faces, 2)
    return Decimated(best.positions, best.faces, birth[best.source_vertex])


def transfer_weights(
    weights: NDArray[np.float32],
    source_vertex: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Carry per-vertex weights through decimation, then renormalise.

    Clustering keeps one representative per cell, so this is a gather rather
    than a resampling — and it preserves the partition of unity `WeightField`
    requires.
    """
    transferred = weights[source_vertex].astype(np.float64)
    sums = transferred.sum(axis=1, keepdims=True)
    np.divide(transferred, sums, out=transferred, where=sums > 0)
    return transferred.astype(np.float32)


def normalise(positions: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Centre at the origin and scale to unit height (spec §5).

    Scale is the largest bounding-box extent rather than the Y extent
    specifically: it agrees with height for an upright character and stays
    meaningful for the models in this corpus that are not upright. The factor is
    returned because deformation error is reported normalised by it (§4.1).
    """
    centre = 0.5 * (positions.max(axis=0) + positions.min(axis=0))
    centred = positions - centre
    scale = float(np.abs(centred).max() * 2.0)
    if scale <= 0:
        return centred, 1.0
    return centred / scale, scale
