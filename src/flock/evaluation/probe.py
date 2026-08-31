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
) -> dict[str, IterationCurve]:
    """Trace every metric across `iterations` on the given meshes.

    This is the shape of every V0 result. A single number cannot separate
    "converged" from "still improving" from "drifting slowly", and those three
    are precisely what P1, P2 and P3 ask about — so nothing here returns one.
    """
    import mlx.core as mx
    import numpy as np

    from flock.backends.mlx.cell import weights_of
    from flock.backends.mlx.rollout import rollout_trace
    from flock.domain.dynamics.state import CellState, StaticInputs
    from flock.domain.skinning.lbs import linear_blend_skin
    from flock.domain.skinning.metrics import (
        bleeding_mass,
        deformation_error,
        stability,
        support_metrics,
        weight_l1,
    )
    from flock.domain.skinning.weights import SUPPORT_THRESHOLD, WeightField
    from flock.training.batch import MeshBank, logits_from_weights

    bank = MeshBank(samples)
    static_tree, _ = bank.gather(np.arange(len(samples)))
    static = StaticInputs(**static_tree)
    state = CellState(
        hidden=mx.zeros((len(samples), initial_weights.shape[1], config.hidden_dim)),
        logits=mx.array(logits_from_weights(initial_weights)),
    )
    captured = rollout_trace(params, state, static, iterations, config)

    series: dict[str, list[float]] = {
        name: [] for name in ("weight_l1", "deformation", "dice", "bleeding", "stability")
    }
    previous: dict[int, Any] = {}
    for step in iterations:
        weights = np.array(weights_of(captured[step], config, static))
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
                stabilities.append(stability(previous[index], predicted))
            previous[index] = predicted
        series["weight_l1"].append(float(np.mean(l1s)))
        series["deformation"].append(float(np.mean(deforms)))
        series["dice"].append(float(np.mean(dices)))
        series["bleeding"].append(float(np.mean(bleeds)))
        series["stability"].append(float(np.mean(stabilities)) if stabilities else float("nan"))

    return {
        name: IterationCurve(metric=name, values=tuple(values))
        for name, values in series.items()
    }


def evaluate_repair(run_name: str, probe: ProbeSet, checkpoint: str) -> dict[str, float]:
    """Measure repair after corrupting a region of a converged state (P3).

    Reports both halves of the property: the fraction of induced error resolved
    inside the corrupted region, and the collateral damage outside it. The
    second is what separates repair from simply re-solving the whole mesh.

    Raises:
        NotImplementedError: Implemented at milestone M4.
    """
    raise NotImplementedError("M4: repair evaluation")
