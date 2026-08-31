"""Vertex-to-bone diffusion distances (spec H1, §2.2).

The correction to the brief's weakest assumption. Local rules see Euclidean
distance, so two regions that are close in space but far across the surface —
thigh against thigh, finger against finger, arm against torso — produce
structural bleeding that no amount of local iteration can undo: separation
cannot operate on a metric blind to the mesh topology.

So geodesy is not left to emerge. It is injected here, once, offline, by the
heat method — which also honours the constraint that no geodesic is ever
recomputed during training.
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from numpy.typing import NDArray

SAMPLES_PER_BONE = 5
"""Points along each bone whose nearest surface vertices seed the heat solve."""


def connected_components(
    num_vertices: int,
    faces: NDArray[np.int32],
) -> tuple[int, NDArray[np.int32]]:
    """Component count and per-vertex label for the mesh graph.

    Meshes in this corpus are routinely fragmented — Articulation-XL 2.0 was
    extended specifically to include models with multiple components, and a
    single decimated character can carry 60+ shells. Heat cannot cross a gap, so
    component structure has to be known before any geodesic claim is made.
    """
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(num_vertices, num_vertices),
    )
    count, labels = csgraph.connected_components(graph + graph.T, directed=False)
    return int(count), labels.astype(np.int32)


def _bone_samples(
    heads: NDArray[np.float64],
    tails: NDArray[np.float64],
    count: int,
) -> NDArray[np.float64]:
    """`[J, count, 3]` points sampled along each bone segment."""
    t = np.linspace(0.0, 1.0, count).reshape(1, count, 1)
    return heads[:, None, :] * (1.0 - t) + tails[:, None, :] * t


def vertex_to_bone_distances(
    positions: NDArray[np.float64],
    faces: NDArray[np.int32],
    heads: NDArray[np.float64],
    tails: NDArray[np.float64],
    diffusion_time: float = 1e-3,
) -> NDArray[np.float32]:
    """Diffusion distance from every vertex to every bone, `[V, J]`.

    Each bone is seeded by the surface vertices nearest to points sampled along
    its segment, and the heat method propagates from that seed set across the
    mesh. Bones live inside the volume, so seeding through nearest surface
    vertices is what ties an interior skeleton to an intrinsic surface metric.

    Fragmented meshes are handled explicitly. Seeding only the globally nearest
    vertex leaves every other component unreachable, and the solver returns
    numbers there that look plausible and mean nothing — on a 67-component
    character that collapsed candidate coverage to 0.40, well under gate G0.

    So each component is seeded independently, at its own vertex nearest the
    bone, and the Euclidean gap from the bone to that seed is added back
    afterwards. The resulting distance reads as *cross to this shell, then
    travel across its surface*:

        d(v, bone) = gap(bone -> seed of v's component) + geodesic(seed -> v)

    Within the shell that actually carries the bone the gap is ~0 and this is
    pure geodesy, so the anatomical separation H1 needs is preserved where the
    topology can express it. Across a detached shell it degrades to a Euclidean
    bridge, which is the only signal a gap admits.

    The heat method also returns small negative values near the source; they are
    clamped, both because a negative distance is meaningless and because
    `CandidateTable` rejects one.
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(positions, faces)
    samples = _bone_samples(heads, tails, SAMPLES_PER_BONE)
    num_components, labels = connected_components(len(positions), faces)
    members = [np.flatnonzero(labels == c) for c in range(num_components)]

    distances = np.empty((len(positions), len(heads)), dtype=np.float64)
    for bone in range(len(heads)):
        delta = positions[:, None, :] - samples[bone][None, :, :]
        squared = np.einsum("vsc,vsc->vs", delta, delta)
        to_bone = squared.min(axis=1)

        seeds: list[int] = []
        gap = np.zeros(len(positions), dtype=np.float64)
        for component in members:
            # One seed per sample point, so a long bone is represented along its
            # whole length. Seeding a single vertex would measure distance from
            # one end of the humerus and read the elbow as far from it.
            local = component[np.argmin(squared[component], axis=0)]
            seeds.extend(int(v) for v in np.unique(local))
            gap[component] = np.sqrt(to_bone[component].min())

        distances[:, bone] = solver.compute_distance_multisource(seeds) + gap

    np.maximum(distances, 0.0, out=distances)
    return distances.astype(np.float32)


def euclidean_point_segment_distances(
    positions: NDArray[np.float64],
    heads: NDArray[np.float64],
    tails: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Point-to-segment Euclidean distance and abscissa, both `[V, J]`.

    A pair feature in its own right, and the candidate-selection metric used by
    ablation A6 — the direct test of H1.

    Returns:
        `(distance, t)` where `t` is the clamped abscissa along each bone.
    """
    axis = tails - heads
    length_squared = np.einsum("jc,jc->j", axis, axis)
    safe = np.where(length_squared > 0, length_squared, 1.0)

    offset = positions[:, None, :] - heads[None, :, :]
    t = np.einsum("vjc,jc->vj", offset, axis) / safe
    t = np.clip(t, 0.0, 1.0)
    # A zero-length bone (the root) collapses to its own point.
    t = np.where(length_squared[None, :] > 0, t, 0.0)

    closest = heads[None, :, :] + t[:, :, None] * axis[None, :, :]
    delta = positions[:, None, :] - closest
    distance = np.sqrt(np.einsum("vjc,vjc->vj", delta, delta))
    return distance.astype(np.float32), t.astype(np.float32)
