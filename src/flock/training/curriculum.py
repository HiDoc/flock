"""Corruption curriculum scheduling (spec §3.2).

Levels are mixed within each batch, with the mixture shifting towards harder
levels over training. One constraint holds throughout: every batch keeps a
fraction of *clean* states, whose target is to be left untouched.

That fraction is what stands between the model and failure mode R3 — collapsing
to the identity is a perfectly good strategy when the input is usually already
close to correct, and it scores well on L1 while making repair impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flock.domain.skinning.corruption import CorruptionLevel


@dataclass(frozen=True)
class CurriculumStage:
    """A mixture of corruption levels active over a step range."""

    until_step: int
    mixture: dict[CorruptionLevel, float]


def default_curriculum() -> tuple[CurriculumStage, ...]:
    """The V0 schedule: C0 first, then C1 and C2, clean throughout."""
    return (
        CurriculumStage(
            3_000,
            {CorruptionLevel.C0_GAUSSIAN_LOGITS: 0.7, CorruptionLevel.CLEAN: 0.3},
        ),
        CurriculumStage(
            8_000,
            {
                CorruptionLevel.C0_GAUSSIAN_LOGITS: 0.3,
                CorruptionLevel.C1_LOCAL_PATCH: 0.4,
                CorruptionLevel.CLEAN: 0.3,
            },
        ),
        CurriculumStage(
            10**9,
            {
                CorruptionLevel.C0_GAUSSIAN_LOGITS: 0.1,
                CorruptionLevel.C1_LOCAL_PATCH: 0.4,
                CorruptionLevel.C2_HIERARCHY_TRANSFER: 0.25,
                CorruptionLevel.CLEAN: 0.25,
            },
        ),
    )


def stage_for(stages: tuple[CurriculumStage, ...], step: int) -> CurriculumStage:
    """The stage active at `step`."""
    for stage in stages:
        if step < stage.until_step:
            return stage
    return stages[-1]


def sample_levels(
    stages: tuple[CurriculumStage, ...],
    step: int,
    count: int,
    rng: np.random.Generator,
) -> list[CorruptionLevel]:
    """Draw `count` corruption levels for the current step."""
    mixture = stage_for(stages, step).mixture
    levels = list(mixture)
    probabilities = np.array([mixture[level] for level in levels], dtype=np.float64)
    probabilities /= probabilities.sum()
    return [levels[i] for i in rng.choice(len(levels), size=count, p=probabilities)]
