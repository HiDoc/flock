"""Pose banks by forward kinematics (spec §3.3, §5).

Poses are baked offline into ready-to-use affine bone transforms, so `L_deform`
— the loss that actually counts (H3) — costs one einsum at train time rather
than a kinematics pass.

Train and test banks are drawn from disjoint random streams. `PreprocessedMesh`
validates that they do not overlap, because a shared pose would let the
reference metric quietly report training performance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flock.domain.geometry.skeleton import ROOT_PARENT, Skeleton


def _random_rotations(
    count: int,
    max_degrees: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """`[count, 3, 3]` rotations of bounded angle about random axes."""
    axes = rng.normal(size=(count, 3))
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    angles = rng.uniform(-max_degrees, max_degrees, size=count) * np.pi / 180.0

    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    zero = np.zeros(count)
    cross = np.stack(
        [np.stack([zero, -z, y], axis=-1),
         np.stack([z, zero, -x], axis=-1),
         np.stack([-y, x, zero], axis=-1)],
        axis=-2,
    )
    outer = axes[:, :, None] * axes[:, None, :]
    eye = np.broadcast_to(np.eye(3), (count, 3, 3))
    c = np.cos(angles)[:, None, None]
    s = np.sin(angles)[:, None, None]
    rotations: NDArray[np.float64] = c * eye + s * cross + (1.0 - c) * outer
    return rotations


def sample_pose_bank(
    skeleton: Skeleton,
    num_poses: int,
    max_degrees: float,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """`[P, J, 3, 4]` affine transforms taking rest vertices to posed ones.

    Each joint gets a bounded random rotation; rotations compose down the
    hierarchy and positions follow, so a shoulder twist carries the whole arm
    the way a real rig does. The returned transform for bone `j` is
    `[R_j | p_j - R_j r_j]`, which is exactly what LBS blends.
    """
    num_bones = skeleton.num_bones
    rest = np.asarray(skeleton.tails, dtype=np.float64)
    parents = np.asarray(skeleton.parents, dtype=np.int64)
    order = np.argsort(skeleton.depths(), kind="stable")

    transforms = np.zeros((num_poses, num_bones, 3, 4), dtype=np.float64)
    for pose in range(num_poses):
        local = _random_rotations(num_bones, max_degrees, rng)
        world = np.zeros((num_bones, 3, 3))
        posed = np.zeros((num_bones, 3))
        for j in order:
            parent = int(parents[j])
            if parent == ROOT_PARENT:
                world[j] = local[j]
                posed[j] = rest[j]
            else:
                world[j] = world[parent] @ local[j]
                posed[j] = posed[parent] + world[parent] @ (rest[j] - rest[parent])
        transforms[pose, :, :, :3] = world
        transforms[pose, :, :, 3] = posed - np.einsum("jab,jb->ja", world, rest)
    return transforms.astype(np.float32)
