"""Anti-corruption layer (spec §5).

Foreign datasets each bring their own axis conventions, bone naming, unit scale
and weight layout. Adapters translate all of it into our value objects exactly
once, at the boundary. Nothing downstream ever learns that Mixamo exists.

Adding a dataset means adding one adapter — which is the entire cost of the V1
RigNet benchmark (spec O4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from flock.domain.geometry.skeleton import Skeleton


class RawRig(Protocol):
    """A mesh plus skeleton plus weights, in whatever form the source uses."""

    @property
    def name(self) -> str:
        """Source-side identifier."""
        ...


class DatasetAdapter(Protocol):
    """Translates one external dataset into our domain types."""

    tier: str
    """Which dataset tier this adapter feeds: `"d1"` or `"d2"`."""

    def discover(self, root: Path) -> list[Path]:
        """List importable rig files under `root`."""
        ...

    def load(self, path: Path) -> RawRig:
        """Read one rig, without interpreting it."""
        ...

    def to_skeleton(self, raw: RawRig) -> Skeleton:
        """Convert to our skeleton: our axis convention, our root, our units."""
        ...
