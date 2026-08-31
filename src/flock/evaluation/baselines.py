"""Baselines B0-B5 (spec §4.2).

Two of these carry the argument.

B3 — a feed-forward GNN, 8 *unshared* layers, same width, roughly 8x the
parameters — is the scientific control for P1. Beating it at equal quality with
a fraction of the parameters is what "the recurrence is useful" means.

B5 — Reynolds' three rules, hand-coded, hand-tuned, zero learning — is the
necessity test for the entire theoretical framing (spec §0.2). If three fixed
rules suffice, learning the cell has no justification. The project's own
grounding supplies the instrument that could sink it, which is the right way
round.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BaselineKind(StrEnum):
    """The six baselines of spec §4.2."""

    B0_INITIALISATION = "b0"
    """The input itself, unrefined. The absolute floor — not beating it is an
    immediate failure."""

    B1_INVERSE_DISTANCE = "b1"
    """Normalised inverse-square distance over candidates: the standard
    geometric heuristic."""

    B2_PER_VERTEX_MLP = "b2"
    """Per-vertex MLP, no messages, equal budget. Do neighbours matter at all?"""

    B3_FEEDFORWARD_GNN = "b3"
    """8 unshared layers, same width, ~8x parameters. The control for P1."""

    B4_SINGLE_STEP = "b4"
    """The trained cell at T=1. Recurrence versus a single pass."""

    B5_HANDCODED_FLOCKING = "b5"
    """Reynolds' three rules iterated: alignment as weight diffusion between
    neighbours, cohesion as attraction to bones near in diffusion distance,
    separation as geodesic attenuation. Step size tuned by grid search."""


@dataclass(frozen=True)
class ReynoldsWeights:
    """Mixing coefficients for B5, set by grid search rather than learned."""

    separation: float
    alignment: float
    cohesion: float
    step: float


def run_baseline(kind: BaselineKind, mesh_id: int, iterations: int) -> dict[str, float]:
    """Evaluate one baseline on one mesh, reporting metrics by iteration.

    Raises:
        NotImplementedError: Implemented at milestone M4.
    """
    raise NotImplementedError("M4: baselines")
