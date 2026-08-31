"""Pose bank value object (spec §3.3, §5).

Poses are precomputed offline by forward kinematics and stored as ready-to-use
bone transforms, so `L_deform` costs one einsum at train time (spec §3.5).
Train and test poses live in separate banks — the anti-leakage rule of §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flock.domain.errors import InvariantError


@dataclass(frozen=True)
class PoseBank:
    """Precomputed bone transforms for a single mesh.

    Attributes:
        transforms: `[P, J, 3, 4]` affine bone transforms in rest-to-posed form.
        split: Either `"train"` or `"test"`; the two must never be mixed.
    """

    transforms: NDArray[np.float32]
    split: str

    def __post_init__(self) -> None:
        """Validate shape and split label."""
        if self.transforms.ndim != 4 or self.transforms.shape[2:] != (3, 4):
            raise InvariantError(f"transforms must be [P, J, 3, 4], got {self.transforms.shape}")
        if self.transforms.shape[0] == 0:
            raise InvariantError("pose bank is empty")
        if self.split not in ("train", "test"):
            raise InvariantError(f"split must be 'train' or 'test', got {self.split!r}")
        if not np.isfinite(self.transforms).all():
            raise InvariantError("transforms contain non-finite values")

    @property
    def num_poses(self) -> int:
        """Number of poses P."""
        return int(self.transforms.shape[0])

    @property
    def num_bones(self) -> int:
        """Number of bones J."""
        return int(self.transforms.shape[1])
