"""Skeleton value object (spec §2.2, §7.3).

The hierarchy is *given* in V0 and V1 — but validated rather than assumed, since
V2 will predict it (spec §7.4) and the tree-validity check is the same one the
benchmark reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flock.domain.errors import InvariantError

ROOT_PARENT = -1


@dataclass(frozen=True)
class Skeleton:
    """A bone hierarchy with head/tail positions in mesh space.

    Attributes:
        heads: `[J, 3]` bone start positions.
        tails: `[J, 3]` bone end positions.
        parents: `[J]` parent bone index, `ROOT_PARENT` for the single root.
        names: Bone names, in index order.
    """

    heads: NDArray[np.float32]
    tails: NDArray[np.float32]
    parents: NDArray[np.int32]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate shapes and that the parent array really is a tree."""
        j = self.heads.shape[0]
        if self.heads.shape != (j, 3) or self.tails.shape != (j, 3):
            raise InvariantError(f"heads/tails must be [J, 3], got {self.heads.shape}")
        if self.parents.shape != (j,):
            raise InvariantError(f"parents must be [J], got {self.parents.shape}")
        if len(self.names) != j:
            raise InvariantError(f"expected {j} names, got {len(self.names)}")
        if j == 0:
            raise InvariantError("skeleton has no bones")

        roots = np.flatnonzero(self.parents == ROOT_PARENT)
        if roots.size != 1:
            raise InvariantError(f"expected exactly one root, found {roots.size}")

        non_root = self.parents[self.parents != ROOT_PARENT]
        if non_root.size and (non_root.min() < 0 or non_root.max() >= j):
            raise InvariantError("parent index out of range")

        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        """Walk every bone to the root; a cycle exceeds J steps."""
        j = self.num_bones
        for start in range(j):
            node, steps = start, 0
            while node != ROOT_PARENT:
                node = int(self.parents[node])
                steps += 1
                if steps > j:
                    raise InvariantError(f"parent cycle reachable from bone {start}")

    @property
    def num_bones(self) -> int:
        """Number of bones J."""
        return int(self.heads.shape[0])

    @property
    def root(self) -> int:
        """Index of the root bone."""
        return int(np.flatnonzero(self.parents == ROOT_PARENT)[0])

    def depths(self) -> NDArray[np.int32]:
        """Hierarchy depth per bone, root at 0 (a §2.2 pair feature)."""
        depth = np.zeros(self.num_bones, dtype=np.int32)
        for b in range(self.num_bones):
            node, d = int(self.parents[b]), 0
            while node != ROOT_PARENT:
                node = int(self.parents[node])
                d += 1
            depth[b] = d
        return depth
