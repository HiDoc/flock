"""Ablations A1-A9 (spec §4.3).

A1, A2, A4 and A6 are the vital subset — the minimum that decides P1 and H1.
The rest can follow once the verdict is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Ablation(StrEnum):
    """The nine ablations of spec §4.3."""

    A1_ITERATION_SWEEP = "a1"
    """T in {1, 2, 4, 8, 16, 32} at eval. Is the gain monotone? (P1, P2) VITAL."""

    A2_WEIGHT_SHARING = "a2"
    """Shared versus unshared parameters: iterating one rule, or raw depth? VITAL."""

    A4_DELTA_VS_ABSOLUTE = "a4"
    """Damped deltas versus predicting logits outright. The core design choice. VITAL."""

    A6_DIFFUSION_FEATURES = "a6"
    """Diffusion-based versus Euclidean candidates. The direct test of H1. VITAL."""

    A3_STATE_POOL = "a3"
    """With and without the pool. Is the pool what creates the attractor?"""

    A5_BONE_CHANNEL = "a5"
    """Bone channel on/off. Where does long-range information come from? (H2)"""

    A7_STOCHASTIC_UPDATE = "a7"
    """Fire rate 0.5 on/off. Attractor robustness."""

    A8_DEFORMATION_LOSS = "a8"
    """L_deform on/off. The test of H3."""

    A9_HIDDEN_WIDTH = "a9"
    """H in {16, 32, 64}. Capacity headroom."""


VITAL: tuple[Ablation, ...] = (
    Ablation.A1_ITERATION_SWEEP,
    Ablation.A2_WEIGHT_SHARING,
    Ablation.A4_DELTA_VS_ABSOLUTE,
    Ablation.A6_DIFFUSION_FEATURES,
)


@dataclass(frozen=True)
class AblationRun:
    """One ablation arm: a config override and the run it produced."""

    ablation: Ablation
    arm: str
    overrides: dict[str, object]
    run_name: str


def plan_ablation(ablation: Ablation, base_run: str) -> list[AblationRun]:
    """Expand an ablation into its arms.

    One change per run, three seeds per arm before any conclusion (spec §8).

    Raises:
        NotImplementedError: Implemented at milestone M4.
    """
    raise NotImplementedError("M4: ablation planning")
