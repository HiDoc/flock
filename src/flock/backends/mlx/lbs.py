"""Fused linear blend skinning (spec §3.5).

One einsum over precomputed `[P, J, 3, 4]` transforms. This is what makes
`L_deform` — the loss that actually matters (H3) — essentially free, which is
why there is no excuse for training on weight L1 alone.

Must agree with the NumPy reference in `flock.domain.skinning.lbs`; the M2
cross-check asserts it.
"""

from __future__ import annotations

import mlx.core as mx


def lbs(positions: mx.array, weights: mx.array, bones: mx.array, transforms: mx.array) -> mx.array:
    """Deform rest positions under bone transforms.

    Args:
        positions: `[N, V, 3]` rest positions.
        weights: `[N, V, B]` weights over candidate slots.
        bones: `[N, V, B]` slot-to-bone indices.
        transforms: `[N, P, J, 3, 4]` affine bone transforms.

    Returns:
        `[N, P, V, 3]` posed positions.
    """
    batch, poses, num_bones = transforms.shape[0], transforms.shape[1], transforms.shape[2]
    vertices, candidates = bones.shape[1], bones.shape[2]

    # Gather the transform of every candidate bone, per pose. Gather only.
    offset = (mx.arange(batch, dtype=mx.int32) * num_bones).reshape(batch, 1, 1)
    flat_bones = (bones + offset).reshape(-1)
    per_pose = mx.take(
        transforms.transpose(1, 0, 2, 3, 4).reshape(poses, batch * num_bones, 3, 4),
        flat_bones,
        axis=1,
    ).reshape(poses, batch, vertices, candidates, 3, 4)

    rotated = mx.einsum("pnvbij,nvj->pnvbi", per_pose[..., :3], positions)
    moved = rotated + per_pose[..., 3]
    blended = mx.einsum("nvb,pnvbi->npvi", weights, moved)
    return blended
