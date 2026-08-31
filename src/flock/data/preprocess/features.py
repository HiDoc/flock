"""Precomputed features (spec §2.2, §1.3).

All features are *invariant* rather than equivariant: distances, dot products,
and abscissae along bones. Combined with random-rotation augmentation, that
buys rotation robustness without the cost of a strictly equivariant
architecture — which becomes necessary only when predicting positions, i.e.
FlockRig in V1 (spec §7.3 O1).

Nothing here reads the ground truth. That is not a convention: the pipeline
computes features and candidates first and admits weights only afterwards
(§5, R7).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

VERTEX_FEATURE_DIM = 5
"""Three covariance eigenvalues of the local neighbourhood, mean neighbour
normal agreement, local area."""

EDGE_FEATURE_DIM = 4
"""Length, dot(n_u, n_v), dot(dir_uv, n_v), dot(dir_uv, n_u)."""

PAIR_FEATURE_DIM = 7
"""Euclidean distance, diffusion distance, abscissa t along the bone,
normal-to-axis angle, bone length, relative hierarchy depth, candidate rank."""


def vertex_features(
    positions: NDArray[np.float64],
    normals: NDArray[np.float64],
    neighbours: NDArray[np.int32],
    neighbour_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Per-vertex invariants, `[V, VERTEX_FEATURE_DIM]`."""
    gathered = positions[neighbours] - positions[:, None, :]
    weight = neighbour_mask[:, :, None].astype(np.float64)
    gathered = gathered * weight
    counts = np.maximum(neighbour_mask.sum(axis=1, keepdims=True), 1)

    covariance = np.einsum("vka,vkb->vab", gathered, gathered) / counts[:, :, None]
    eigenvalues = np.linalg.eigvalsh(covariance)[:, ::-1]
    total = np.maximum(eigenvalues.sum(axis=1, keepdims=True), 1e-12)
    shape = eigenvalues / total  # scale-free: curvature-like, not size-like

    agreement = np.einsum("vkc,vc->vk", normals[neighbours], normals)
    mean_agreement = (agreement * neighbour_mask).sum(axis=1, keepdims=True) / counts

    area = np.sqrt(np.einsum("vkc,vkc->vk", gathered, gathered)).sum(axis=1, keepdims=True) / counts
    return np.concatenate([shape, mean_agreement, area], axis=1).astype(np.float32)


def edge_features(
    positions: NDArray[np.float64],
    normals: NDArray[np.float64],
    neighbours: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Per-neighbour invariants, `[V, K, EDGE_FEATURE_DIM]`."""
    delta = positions[neighbours] - positions[:, None, :]
    length = np.sqrt(np.einsum("vkc,vkc->vk", delta, delta))
    direction = delta / np.maximum(length, 1e-12)[:, :, None]

    normal_agreement = np.einsum("vkc,vc->vk", normals[neighbours], normals)
    along_neighbour = np.einsum("vkc,vkc->vk", direction, normals[neighbours])
    along_self = np.einsum("vkc,vc->vk", direction, normals)
    return np.stack([length, normal_agreement, along_neighbour, along_self], axis=-1).astype(
        np.float32
    )


def pair_features(
    normals: NDArray[np.float64],
    heads: NDArray[np.float64],
    tails: NDArray[np.float64],
    depths: NDArray[np.int32],
    candidate_bones: NDArray[np.int32],
    euclidean: NDArray[np.float32],
    abscissa: NDArray[np.float32],
    diffusion: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Vertex-to-candidate-bone features, `[V, B, PAIR_FEATURE_DIM]`."""
    rows = np.arange(len(candidate_bones))[:, None]
    axis = tails - heads
    bone_length = np.linalg.norm(axis, axis=1)
    direction = axis / np.maximum(bone_length, 1e-12)[:, None]

    angle = np.einsum("vc,vbc->vb", normals, direction[candidate_bones])
    relative_depth = depths[candidate_bones].astype(np.float64)
    relative_depth = relative_depth - relative_depth.min(axis=1, keepdims=True)
    rank = np.broadcast_to(
        np.arange(candidate_bones.shape[1], dtype=np.float64), candidate_bones.shape
    )

    return np.stack(
        [
            euclidean[rows, candidate_bones],
            diffusion[rows, candidate_bones],
            abscissa[rows, candidate_bones],
            angle,
            bone_length[candidate_bones],
            relative_depth,
            rank,
        ],
        axis=-1,
    ).astype(np.float32)
