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
from typing import Any

import numpy as np
from numpy.typing import NDArray


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


DEFAULT_REYNOLDS = ReynoldsWeights(separation=1.0, alignment=0.5, cohesion=1.0, step=0.3)
"""Grid-searched starting point for B5; `tune_reynolds` refines it per dataset."""


def inverse_distance_weights(sample: Any) -> NDArray[np.float32]:
    """B1: normalised inverse-square distance over the candidates.

    The standard geometric heuristic and the floor any learned model has to
    clear. It reads only the candidate distances — no training, no ground truth.
    """
    distances = np.maximum(sample.candidates.distances.astype(np.float64), 1e-6)
    weights = sample.candidates.mask / (distances**2)
    total = weights.sum(axis=1, keepdims=True)
    np.divide(weights, total, out=weights, where=total > 0)
    normalised: NDArray[np.float32] = weights.astype(np.float32)
    return normalised


def reynolds_step(
    weights: NDArray[np.float32],
    sample: Any,
    coefficients: ReynoldsWeights,
) -> NDArray[np.float32]:
    """One iteration of Reynolds' three rules, hand-coded (B5).

    * **alignment** — diffuse each vertex's weights towards its neighbours', the
      direct analogue of matching a neighbour's heading;
    * **cohesion** — pull mass towards candidates that are near in diffusion
      distance, the attraction term;
    * **separation** — attenuate mass on candidates that are far, the repulsion
      that keeps geodesically distant bones out.

    Zero learning: three fixed rules and a step size. If this suffices, learning
    the cell is not justified (spec §0.2), which is what makes B5 the necessity
    test for the whole framing rather than a courtesy comparison.
    """
    neighbours = sample.mesh.neighbours
    mask = sample.mesh.neighbour_mask[:, :, None]
    live = np.maximum(mask.sum(axis=1), 1.0)
    alignment = (weights[neighbours] * mask).sum(axis=1) / live

    distances = np.maximum(sample.candidates.distances.astype(np.float64), 1e-6)
    near = sample.candidates.mask / (distances**2)
    total = near.sum(axis=1, keepdims=True)
    np.divide(near, total, out=near, where=total > 0)

    scale = np.median(distances[sample.candidates.mask]) if sample.candidates.mask.any() else 1.0
    separation = np.exp(-distances / max(scale, 1e-6)) * sample.candidates.mask

    target = (
        coefficients.alignment * alignment
        + coefficients.cohesion * near
        + coefficients.separation * weights * separation
    )
    updated = (1.0 - coefficients.step) * weights + coefficients.step * target
    updated = np.clip(updated, 0.0, None) * sample.candidates.mask
    total = updated.sum(axis=1, keepdims=True)
    np.divide(updated, total, out=updated, where=total > 0)
    stepped: NDArray[np.float32] = updated.astype(np.float32)
    return stepped


def run_reynolds(
    sample: Any,
    initial: NDArray[np.float32],
    iterations: int,
    coefficients: ReynoldsWeights = DEFAULT_REYNOLDS,
) -> NDArray[np.float32]:
    """Iterate B5's fixed rules `iterations` times."""
    weights = initial.astype(np.float32)
    for _ in range(iterations):
        weights = reynolds_step(weights, sample, coefficients)
    return weights


def tune_reynolds(
    samples: list[Any],
    initials: NDArray[np.float32],
    iterations: int = 8,
    seed: int = 0,
) -> tuple[ReynoldsWeights, float]:
    """Grid-search B5's coefficients, as spec §4.2 requires.

    An untuned hand-coded baseline is a straw man; the comparison only means
    something if the fixed rules were given their best shot.
    """
    from flock.domain.skinning.metrics import weight_l1
    from flock.domain.skinning.weights import WeightField

    # The grid reaches well below the obvious values. A coarse grid that only
    # offers strong coefficients would let B5 lose to its own step size rather
    # than to the hypothesis, and "three fixed rules do not suffice" is only
    # worth saying if the rules were given their best shot (spec §4.2).
    best, best_score = DEFAULT_REYNOLDS, float("inf")
    grid = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    for separation in grid:
        for alignment in grid:
            for cohesion in grid:
                for step in (0.02, 0.05, 0.1, 0.3, 0.5):
                    candidate = ReynoldsWeights(separation, alignment, cohesion, step)
                    score = float(np.mean([
                        weight_l1(
                            WeightField(run_reynolds(s, initials[i], iterations, candidate)),
                            s.ground_truth,
                        )
                        for i, s in enumerate(samples)
                    ]))
                    if score < best_score:
                        best, best_score = candidate, score
    return best, best_score
