"""Linear blend skinning (spec §3.3, §3.5).

Reference NumPy implementation, used by tests and offline tooling. The training
path uses the backend's fused version; the two must agree, which is what the
M2 cross-check asserts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flock.domain.skinning.weights import CandidateTable, WeightField


def linear_blend_skin(
    positions: NDArray[np.float32],
    weights: WeightField,
    candidates: CandidateTable,
    transforms: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Deform rest positions under precomputed bone transforms.

    Args:
        positions: `[V, 3]` rest vertex positions.
        weights: Weights over candidate slots.
        candidates: Slot-to-bone mapping.
        transforms: `[P, J, 3, 4]` affine bone transforms.

    Returns:
        `[P, V, 3]` posed vertex positions.
    """
    bones = candidates.bones
    rotations = transforms[:, :, :, :3][:, bones]      # [P, V, B, 3, 3]
    translations = transforms[:, :, :, 3][:, bones]    # [P, V, B, 3]
    moved = np.einsum("pvbij,vj->pvbi", rotations, positions) + translations
    posed = np.einsum("vb,pvbi->pvi", weights.values, moved)
    return np.asarray(posed, dtype=np.float32)
