"""Mesh neighbourhood construction (spec §2.2, §3.5).

The one-ring capped at K=8, topped up by nearest neighbours where valence falls
short. Fixed `[V, K]` indices with an explicit mask, because aggregation must be
a gather over constant tables — the only access pattern MLX/Metal handles well,
and the one that keeps shapes constant for the compiler.
"""

from __future__ import annotations

import igl
import numpy as np
from numpy.typing import NDArray


def build_neighbours(
    positions: NDArray[np.float64],
    faces: NDArray[np.int32],
    num_neighbours: int,
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """`([V, K] indices, [V, K] mask)` one-ring topped up by KNN.

    A vertex with more than K one-ring neighbours keeps the K nearest; one with
    fewer is filled from its nearest vertices by Euclidean distance, so every
    row is usable and the mask marks only genuinely empty slots.
    """
    count = len(positions)
    adjacency = igl.adjacency_list(faces.astype(np.int64))
    indices = np.zeros((count, num_neighbours), dtype=np.int32)
    mask = np.zeros((count, num_neighbours), dtype=bool)

    for v in range(count):
        ring = [int(u) for u in adjacency[v]] if v < len(adjacency) else []
        if len(ring) > num_neighbours:
            delta = positions[ring] - positions[v]
            nearest = np.argsort(np.einsum("nc,nc->n", delta, delta))[:num_neighbours]
            ring = [ring[i] for i in nearest]
        elif len(ring) < num_neighbours:
            delta = positions - positions[v]
            order = np.argsort(np.einsum("nc,nc->n", delta, delta))
            for u in order:
                if len(ring) >= num_neighbours:
                    break
                if int(u) != v and int(u) not in ring:
                    ring.append(int(u))
        take = min(len(ring), num_neighbours)
        indices[v, :take] = ring[:take]
        mask[v, :take] = True
    return indices, mask
