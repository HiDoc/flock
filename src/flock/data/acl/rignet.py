"""D2 — RigNet / ModelsResource adapter (spec §5, §7.3 O4).

2703 models with standard splits. Reserved for V1: it is the comparison
benchmark, and comparing before P1-P3 are settled would answer a question V0
has not yet earned the right to ask (spec H7).
"""

from __future__ import annotations

from pathlib import Path

from flock.data.acl.base import RawRig
from flock.domain.geometry.skeleton import Skeleton


class RigNetAdapter:
    """Reads the RigNet dataset into domain types."""

    tier = "d2"

    def discover(self, root: Path) -> list[Path]:
        """List RigNet models under `root`.

        Raises:
            NotImplementedError: Implemented at milestone V1-O4.
        """
        raise NotImplementedError("V1 (O4): RigNet discovery")

    def load(self, path: Path) -> RawRig:
        """Read one RigNet model.

        Raises:
            NotImplementedError: Implemented at milestone V1-O4.
        """
        raise NotImplementedError("V1 (O4): RigNet load")

    def to_skeleton(self, raw: RawRig) -> Skeleton:
        """Convert a RigNet hierarchy to ours.

        Raises:
            NotImplementedError: Implemented at milestone V1-O4.
        """
        raise NotImplementedError("V1 (O4): RigNet skeleton conversion")
