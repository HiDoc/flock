"""Quality metrics (spec §4.1, §4.4).

Every metric here is traced as a function of iteration count T — that curve, not
a single number, is what decides P1 and P2. Deformation error is the reference
metric; weight L1 is a control only (spec H3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flock.domain.skinning.weights import CandidateTable, WeightField


@dataclass(frozen=True)
class SupportMetrics:
    """Support agreement at a declared threshold (spec §4.1)."""

    precision: float
    recall: float
    dice: float
    mean_influences: float
    threshold: float


def weight_l1(predicted: WeightField, target: WeightField) -> float:
    """Mean L1 distance between weight rows.

    Control metric only. Artist weights are non-unique, so L1 punishes correct
    solutions and rewards memorising conventions (spec H3).

    Returns:
        Mean absolute weight difference per vertex.
    """
    return float(np.abs(predicted.values - target.values).sum(axis=1).mean())


def deformation_error(
    posed_predicted: NDArray[np.float32],
    posed_target: NDArray[np.float32],
    mesh_height: float,
) -> float:
    """Height-normalised L2 vertex error under test poses.

    The reference metric: it measures what a rig is actually for. Artist weights
    are non-unique, so two very different weight fields can deform identically —
    which is exactly why L1 is the control and this is the number that counts
    (spec H3).

    Args:
        posed_predicted: `[P, V, 3]` vertices posed with the predicted weights.
        posed_target: `[P, V, 3]` vertices posed with the ground truth.
        mesh_height: Normalisation scale, so the metric is comparable across meshes.

    Returns:
        Mean per-vertex displacement error, in mesh heights.
    """
    error = np.linalg.norm(posed_predicted - posed_target, axis=-1)
    return float(error.mean() / max(mesh_height, 1e-9))


def support_metrics(
    predicted: WeightField,
    target: WeightField,
    threshold: float,
) -> SupportMetrics:
    """Precision, recall, Dice and mean influence count over the support.

    The threshold is reported rather than hidden: softmax emits no exact zeros,
    so every support number is a function of where the cut is placed (spec H5).
    """
    predicted_support = predicted.support(threshold)
    target_support = target.support(threshold)
    overlap = float((predicted_support & target_support).sum())
    predicted_count = float(predicted_support.sum())
    target_count = float(target_support.sum())

    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / target_count if target_count else 0.0
    denominator = predicted_count + target_count
    return SupportMetrics(
        precision=precision,
        recall=recall,
        dice=(2.0 * overlap / denominator) if denominator else 0.0,
        mean_influences=float(predicted_support.sum(axis=1).mean()),
        threshold=threshold,
    )


def bleeding_mass(
    predicted: WeightField,
    candidates: CandidateTable,
    max_distance: float,
) -> float:
    """Weight mass carried by geodesically distant bones.

    Defined against diffusion distance rather than the GT support, so it stays
    meaningful when the ground truth itself is a convention (spec §4.1). This is
    the direct measure of the failure H1 predicts: weight leaking onto a bone
    that is close in space but far across the surface.
    """
    distant = candidates.distances > max_distance
    return float((predicted.values * distant * candidates.mask).sum(axis=1).mean())


def coverage(
    dense_weights: NDArray[np.float32],
    candidates: CandidateTable,
) -> NDArray[np.float32]:
    """Per-vertex GT weight mass reachable within the candidate set (spec §4.4).

    The gate-G0 diagnostic: `coverage(v) = sum over b in cand(v) of w_GT(v, b)`.

    It must be measured against the **full per-bone** ground truth, before the
    weights are restricted to candidate slots and renormalised. Measured after,
    every row sums to 1 by construction and the metric would report a perfect
    score while saying nothing — the exact silent failure R1 describes.

    Args:
        dense_weights: `[V, J]` ground-truth weights over every skeleton bone.
        candidates: The candidate table to score.

    Returns:
        `[V]` coverage per vertex.
    """
    rows = np.arange(dense_weights.shape[0])[:, None]
    gathered = dense_weights[rows, candidates.bones]
    return (gathered * candidates.mask).sum(axis=1).astype(np.float32)


def stability(weights_t: WeightField, weights_next: WeightField) -> float:
    """Median per-vertex L1 change between consecutive iterations (spec §4.1).

    The attractor signal for P2: this should decay towards zero and stay there.
    Median rather than mean, so a handful of oscillating vertices cannot be
    averaged into the appearance of convergence.

    Measured between two states, whatever the gap between them. Callers
    reporting it over a non-uniform grid of T must divide by that gap, or a
    constant rate reads as divergence (see `evaluate_by_iteration`).
    """
    delta = np.abs(weights_next.values - weights_t.values).sum(axis=1)
    return float(np.median(delta))


@dataclass(frozen=True)
class RepairReport:
    """Repair of a damaged region, and the harm done outside it (spec §4.1, P3).

    Both halves are needed, and neither is meaningful alone. A cell that simply
    re-solves the whole mesh from scratch scores perfectly on repair while
    destroying the correct 89% around the damage — which is risk R3, the
    do-no-harm failure, and the reason gate G3 has two thresholds.

    Attributes:
        repaired_fraction: Share of the *induced* error resolved inside the
            region. 1.0 means the state returned to the quality it had before
            the damage; above 1.0 means it came back better, which is possible
            because the pre-damage state was not itself perfect.
        collateral_relative: Relative change in error outside the region.
            Positive is damage, negative is incidental improvement.
        induced_error: Error the damage added inside the region, the
            denominator of `repaired_fraction`. Reported so a near-zero
            denominator is visible rather than inferred from a wild ratio.
        residual_error: Error still standing inside the region afterwards.
    """

    repaired_fraction: float
    collateral_relative: float
    induced_error: float
    residual_error: float


def _row_l1(predicted: WeightField, target: WeightField) -> NDArray[np.float64]:
    """Per-vertex L1 between two weight fields."""
    return np.abs(predicted.values - target.values).sum(axis=1).astype(np.float64)


def repair_scores(
    settled: WeightField,
    damaged: WeightField,
    repaired: WeightField,
    target: WeightField,
    region: NDArray[np.bool_],
    vertex_mask: NDArray[np.bool_],
) -> RepairReport:
    """Score one repair episode against gate G3.

    Error is measured against the ground truth throughout, not against the
    settled state: returning precisely to a converged-but-wrong state is not
    repair. `settled` sets the bar the dynamics had already reached, so the
    fraction reported is of the error the damage *added* rather than of all
    error present.

    Args:
        settled: The converged state, before damage.
        damaged: The same state with the region corrupted.
        repaired: The state after the dynamics ran on from the damage.
        target: Ground truth.
        region: `[V]` mask of the damaged region Omega.
        vertex_mask: `[V]` mask of real (non-padding) vertices.

    Returns:
        The report. `repaired_fraction` is NaN when the damage induced no error
        inside the region — an episode that tests nothing, which the caller is
        expected to drop rather than score.
    """
    inside = region & vertex_mask
    outside = (~region) & vertex_mask

    before = _row_l1(damaged, target)
    after = _row_l1(repaired, target)
    baseline = _row_l1(settled, target)

    induced = float(before[inside].mean() - baseline[inside].mean()) if inside.any() else 0.0
    residual = float(after[inside].mean() - baseline[inside].mean()) if inside.any() else 0.0
    fraction = float("nan") if induced <= 1e-9 else (induced - residual) / induced

    if outside.any():
        undamaged = float(baseline[outside].mean())
        collateral = (float(after[outside].mean()) - undamaged) / max(undamaged, 1e-9)
    else:
        collateral = 0.0

    return RepairReport(
        repaired_fraction=fraction,
        collateral_relative=collateral,
        induced_error=induced,
        residual_error=residual,
    )
