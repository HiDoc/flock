"""D0 — procedural dataset (spec §5).

Capsule chains and trees with an *analytic* ground truth
`w ∝ softmax(−d_diff / σ)`. Because the GT is known in closed form, D0 gives
perfect falsifiability: any gap between model and target is the model's, not the
data's. Difficulty is dialled explicitly — σ, joint angles, and deliberate
proximity between limbs, which is the configuration that makes H1 (anatomical
separation) bite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flock.domain.sample import PreprocessedMesh


@dataclass(frozen=True)
class D0Config:
    """Generation parameters for one procedural rig."""

    num_bones: int = 5
    """3-8 in the standard sweep."""

    sigma: float = 0.15
    """Falloff of the analytic weight function; smaller means sharper."""

    limb_proximity: float = 0.0
    """0 keeps limbs apart; higher values place them close in Euclidean space
    while staying far geodesically — the hard case for candidate selection."""

    branching: bool = False
    """False builds a chain, True a tree."""

    rings: int = 24
    """Tube cross-section resolution before decimation."""


def generate(config: D0Config, seed: int, mesh_id: int) -> PreprocessedMesh:
    """Generate one procedural rig with analytic ground-truth weights.

    Raises:
        NotImplementedError: Implemented at milestone M1.
    """
    raise NotImplementedError("M1: D0 generator")


def default_suite(count: int, rng: np.random.Generator) -> list[D0Config]:
    """A spread of configs from easy chains to close-limbed trees.

    Raises:
        NotImplementedError: Implemented at milestone M1.
    """
    raise NotImplementedError("M1: D0 suite")
