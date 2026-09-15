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


def confident_error_curriculum() -> tuple[CurriculumStage, ...]:
    """`default_curriculum` with half of C1's share spent on C1P.

    The one change that tests the ADR-0008 finding: every corruption the default
    curriculum asks the model to repair arrives *flat*, so peakedness is a
    near-perfect proxy for "needs work" and the model learns that instead of
    learning to judge correctness. C1P is the same patch size and the same
    budget, but peaked — confidence stops being evidence of being right.

    C1's total share is unchanged, so the arm differs from the default in what
    the corruption looks like and in nothing else.
    """
    swapped = []
    for stage in default_curriculum():
        mixture = dict(stage.mixture)
        share = mixture.pop(CorruptionLevel.C1_LOCAL_PATCH, 0.0)
        if share:
            mixture[CorruptionLevel.C1_LOCAL_PATCH] = share / 2
            mixture[CorruptionLevel.C1P_PERMUTED_PATCH] = share / 2
        swapped.append(CurriculumStage(stage.until_step, mixture))
    return tuple(swapped)


def with_clean_share(
    stages: tuple[CurriculumStage, ...], share: float
) -> tuple[CurriculumStage, ...]:
    """Set every stage's clean share to `share`, moving the difference to C0.

    The lever for [ADR-0010]'s leading reading of why training on confident
    errors failed: clean states are peaked by construction and ~45% of the
    resident pool, so the signal saying *leave peaked states alone* outvotes
    the one saying *fix these peaked states* roughly 3:1. Lowering it tests
    that directly.

    The freed mass goes to C0 rather than being spread, so the share of every
    *other* level is untouched and the clean-to-corrupted ratio is the only
    thing that moves. Note the pool keeps its own `CLEAN_FRACTION` of forced
    clean slots regardless — §3.1's do-no-harm floor is not up for negotiation
    here, so this cannot drive the clean signal to zero.
    """
    if not 0.0 <= share < 1.0:
        raise ValueError(f"clean share must be in [0, 1), got {share}")
    rescaled = []
    for stage in stages:
        mixture = dict(stage.mixture)
        freed = mixture.get(CorruptionLevel.CLEAN, 0.0) - share
        mixture[CorruptionLevel.CLEAN] = share
        mixture[CorruptionLevel.C0_GAUSSIAN_LOGITS] = (
            mixture.get(CorruptionLevel.C0_GAUSSIAN_LOGITS, 0.0) + freed
        )
        rescaled.append(CurriculumStage(stage.until_step, mixture))
    return tuple(rescaled)


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
