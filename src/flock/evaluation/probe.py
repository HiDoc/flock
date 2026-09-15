"""The probe set and metrics-by-iteration (spec §8, §4.1).

Eight fixed meshes — 2 easy procedural, 2 hard procedural with limbs
deliberately close, 4 Mixamo — evaluated identically on every run. Fixed, so
that two runs are actually comparable; small, so that evaluating by iteration is
cheap enough to do every time.

Every metric is a curve over T, never a single number. A single number cannot
distinguish "converged" from "still improving" from "diverging slowly", and
those three are precisely what P1, P2 and P3 are asking about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EVAL_ITERATIONS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
"""The T values every metric is reported at (spec §4.1)."""

BLEEDING_DISTANCE = 0.25
"""Diffusion distance beyond which weight counts as bleeding (spec §4.1)."""

PROBE_SIZE = 8


@dataclass(frozen=True)
class ProbeSet:
    """The fixed evaluation meshes."""

    easy_procedural: tuple[int, ...]
    hard_procedural: tuple[int, ...]
    mixamo: tuple[int, ...]

    @property
    def mesh_ids(self) -> tuple[int, ...]:
        """All probe mesh ids in a stable order."""
        return self.easy_procedural + self.hard_procedural + self.mixamo


@dataclass(frozen=True)
class IterationCurve:
    """One metric traced over `EVAL_ITERATIONS`."""

    metric: str
    values: tuple[float, ...]


def evaluate_by_iteration(
    params: Any,
    samples: list[Any],
    config: Any,
    initial_weights: Any,
    iterations: tuple[int, ...] = EVAL_ITERATIONS,
    backend_name: str = "mlx",
) -> dict[str, IterationCurve]:
    """Trace every metric across `iterations` on the given meshes.

    This is the shape of every V0 result. A single number cannot separate
    "converged" from "still improving" from "drifting slowly", and those three
    are precisely what P1, P2 and P3 ask about — so nothing here returns one.

    The rollout runs behind the backend port; every metric below it is plain
    NumPy over domain value objects, so the numbers are computed identically
    whatever ran the dynamics.
    """
    import numpy as np

    from flock.backends.registry import get_backend
    from flock.domain.skinning.lbs import linear_blend_skin
    from flock.domain.skinning.metrics import (
        bleeding_mass,
        deformation_error,
        stability,
        support_metrics,
        weight_l1,
    )
    from flock.domain.skinning.weights import SUPPORT_THRESHOLD, WeightField
    from flock.training.batch import logits_from_weights

    backend = get_backend(backend_name, config)
    bank = backend.make_bank(samples)
    static_tree, _ = bank.gather(np.arange(len(samples)))
    captured = backend.trace_from(
        params, None, logits_from_weights(initial_weights), static_tree, iterations, config
    )

    series: dict[str, list[float]] = {
        name: []
        for name in ("weight_l1", "deformation", "dice", "bleeding", "stability_per_step")
    }
    previous: dict[int, Any] = {}
    previous_step = 0
    for step in iterations:
        weights = captured[step].weights
        l1s, deforms, dices, bleeds, stabilities = [], [], [], [], []
        for index, sample in enumerate(samples):
            predicted = WeightField(weights[index])
            l1s.append(weight_l1(predicted, sample.ground_truth))
            posed = linear_blend_skin(
                sample.mesh.positions.astype(np.float64), predicted,
                sample.candidates, sample.test_poses.transforms,
            )
            reference = linear_blend_skin(
                sample.mesh.positions.astype(np.float64), sample.ground_truth,
                sample.candidates, sample.test_poses.transforms,
            )
            deforms.append(deformation_error(posed, reference, 1.0))
            dices.append(support_metrics(predicted, sample.ground_truth, SUPPORT_THRESHOLD).dice)
            bleeds.append(bleeding_mass(predicted, sample.candidates, BLEEDING_DISTANCE))
            if index in previous:
                # Per step, not per checkpoint. `iterations` is deliberately
                # non-uniform, so a raw consecutive-checkpoint difference grows
                # with the gap and reads as divergence when the rate is flat.
                stabilities.append(stability(previous[index], predicted) / (step - previous_step))
            previous[index] = predicted
        series["weight_l1"].append(float(np.mean(l1s)))
        series["deformation"].append(float(np.mean(deforms)))
        series["dice"].append(float(np.mean(dices)))
        series["bleeding"].append(float(np.mean(bleeds)))
        series["stability_per_step"].append(
            float(np.mean(stabilities)) if stabilities else float("nan")
        )
        previous_step = step

    return {
        name: IterationCurve(metric=name, values=tuple(values))
        for name, values in series.items()
    }


REPAIR_ITERATIONS: tuple[int, ...] = (0, 1, 2, 4, 8)
"""Iterations after the damage at which repair is scored (gate G3 reads T=8)."""

SETTLE_ITERATIONS = 8
"""Steps run before the damage, so that what gets damaged is a converged state."""


def evaluate_repair(
    params: Any,
    samples: list[Any],
    config: Any,
    level: str = "c1",
    settle: int = SETTLE_ITERATIONS,
    checkpoints: tuple[int, ...] = REPAIR_ITERATIONS,
    seed: int = 0,
    backend_name: str = "mlx",
) -> dict[str, Any]:
    """Measure repair after damaging a region of a converged state (P3, gate G3).

    The protocol is what distinguishes this from the C1 evaluation already run
    at every checkpoint. There, corruption is applied to the ground truth and
    the cell solves from a cold start. Here the cell first runs to convergence,
    and *its own* settled state is then damaged — which is what P3 actually
    claims, and the only version of the question the state pool's recorruption
    cycle is training for.

    The hidden state is zeroed inside the damaged region. Leaving it intact
    would let the cell recover by reading its own memory of the answer, and
    repair driven by memory rather than by neighbours is not the property under
    test.

    Returns:
        The gate metrics at the last checkpoint, plus per-checkpoint curves and
        the number of episodes dropped for inducing no measurable error.
        `drift_relative` is the same complement-region change measured on an
        *undamaged* rollout of equal length: collateral above that line is what
        the damage actually cost.
    """
    import numpy as np

    from flock.backends.registry import get_backend
    from flock.domain.skinning.corruption import CorruptionLevel, corrupt
    from flock.domain.skinning.metrics import repair_scores
    from flock.domain.skinning.weights import WeightField
    from flock.training.batch import logits_from_weights

    backend = get_backend(backend_name, config)
    bank = backend.make_bank(samples)
    static_tree, _ = bank.gather(np.arange(len(samples)))

    ground_truth = np.stack([s.ground_truth.values for s in samples])
    settled = backend.trace_from(
        params, None, logits_from_weights(ground_truth), static_tree, (settle,), config
    )[settle]

    rng = np.random.default_rng(seed)
    damaged = np.empty_like(settled.weights)
    regions = np.zeros(settled.weights.shape[:2], dtype=bool)
    for index, sample in enumerate(samples):
        field, region = corrupt(
            WeightField(settled.weights[index]),
            CorruptionLevel(level),
            sample.corruption_context(),
            rng,
        )
        damaged[index], regions[index] = field.values, region

    hidden = settled.hidden * ~regions[:, :, None]
    traced = backend.trace_from(
        params, hidden, logits_from_weights(damaged), static_tree, checkpoints, config
    )
    # The control that makes the collateral number mean something: the same
    # state, the same number of further steps, no damage. Without it, drift the
    # dynamics would have shown anyway (gate G2) is billed to the damage.
    undamaged = backend.trace_from(
        params, settled.hidden, settled.logits, static_tree, checkpoints, config
    )

    curves: dict[str, list[float]] = {
        "repaired_fraction": [], "collateral_relative": [],
        "drift_relative": [], "moved_inside": [],
    }
    dropped = 0
    for step in checkpoints:
        fractions, collaterals, drifts, movements = [], [], [], []
        for index, sample in enumerate(samples):
            base = WeightField(settled.weights[index])
            report = repair_scores(
                base,
                WeightField(damaged[index]),
                WeightField(traced[step].weights[index]),
                sample.ground_truth,
                regions[index],
                sample.mesh.vertex_mask,
            )
            # An episode whose damage induced no error tests nothing; scoring it
            # would credit the cell for a repair it never had to make.
            if np.isnan(report.repaired_fraction):
                dropped += 1
                continue
            control = repair_scores(
                base, WeightField(damaged[index]),
                WeightField(undamaged[step].weights[index]),
                sample.ground_truth, regions[index], sample.mesh.vertex_mask,
            )
            fractions.append(report.repaired_fraction)
            collaterals.append(report.collateral_relative)
            drifts.append(control.collateral_relative)
            # How far the state moved inside the region, repair or not. This is
            # what separates "tried and failed" from "saw nothing to fix": a
            # corruption that is locally self-consistent is a fixed point, and
            # a repaired_fraction of zero alone cannot tell the two apart.
            inside = regions[index] & sample.mesh.vertex_mask
            movements.append(float(np.abs(
                traced[step].weights[index] - damaged[index]
            ).sum(axis=1)[inside].mean()) if inside.any() else 0.0)
        curves["repaired_fraction"].append(float(np.mean(fractions)) if fractions else float("nan"))
        curves["collateral_relative"].append(
            float(np.mean(collaterals)) if collaterals else float("nan")
        )
        curves["drift_relative"].append(float(np.mean(drifts)) if drifts else float("nan"))
        curves["moved_inside"].append(float(np.mean(movements)) if movements else float("nan"))

    return {
        "repaired_fraction": curves["repaired_fraction"][-1],
        "collateral_relative": curves["collateral_relative"][-1],
        "drift_relative": curves["drift_relative"][-1],
        "moved_inside": curves["moved_inside"][-1],
        "iterations": list(checkpoints),
        "curves": curves,
        "episodes": len(samples),
        "dropped": dropped // max(len(checkpoints), 1),
        "region_fraction": float(
            np.mean([
                regions[i][s.mesh.vertex_mask].mean() for i, s in enumerate(samples)
            ])
        ),
    }
