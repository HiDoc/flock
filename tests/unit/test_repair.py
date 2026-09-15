"""Repair scoring (spec §4.1, gate G3, property P3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flock.domain.geometry.mesh import NUM_CANDIDATES, NUM_VERTICES
from flock.domain.skinning.metrics import repair_scores
from flock.domain.skinning.weights import WeightField


def field(*rows: tuple[float, float]) -> WeightField:
    """A four-vertex weight field padded to the fixed `[V, B]` shape."""
    values = np.zeros((NUM_VERTICES, NUM_CANDIDATES), dtype=np.float32)
    for index, (first, second) in enumerate(rows):
        values[index, 0], values[index, 1] = first, second
    return WeightField(values)


def mask(*live: bool) -> np.ndarray:
    """A `[V]` mask over the first four vertices."""
    values = np.zeros(NUM_VERTICES, dtype=bool)
    values[: len(live)] = live
    return values


TARGET = field((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
SETTLED = field((1.0, 0.0), (1.0, 0.0), (0.1, 0.9), (0.1, 0.9))
DAMAGED = field((0.0, 1.0), (0.0, 1.0), (0.1, 0.9), (0.1, 0.9))
REGION = mask(True, True, False, False)
LIVE = mask(True, True, True, True)


class TestRepairScores:
    """The two halves of G3 must be measurable independently."""

    def test_returning_to_the_settled_state_is_full_repair(self) -> None:
        report = repair_scores(SETTLED, DAMAGED, SETTLED, TARGET, REGION, LIVE)
        assert report.repaired_fraction == 1.0
        assert report.collateral_relative == 0.0
        assert report.induced_error == 2.0

    def test_standing_still_repairs_nothing(self) -> None:
        report = repair_scores(SETTLED, DAMAGED, DAMAGED, TARGET, REGION, LIVE)
        assert report.repaired_fraction == 0.0
        assert report.residual_error == report.induced_error

    def test_damage_outside_the_region_is_collateral(self) -> None:
        # Repaired inside, but the untouched 50% drifted from 0.2 L1 to 0.4.
        harmed = field((1.0, 0.0), (1.0, 0.0), (0.2, 0.8), (0.2, 0.8))
        report = repair_scores(SETTLED, DAMAGED, harmed, TARGET, REGION, LIVE)
        assert report.repaired_fraction == 1.0
        assert report.collateral_relative == pytest.approx(1.0)

    def test_repair_beyond_the_settled_state_exceeds_one(self) -> None:
        # The pre-damage state was not itself perfect, so over-recovery is real
        # and must not be silently clipped to 1.0.
        better = field((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0))
        report = repair_scores(SETTLED, DAMAGED, better, TARGET, REGION, LIVE)
        assert report.collateral_relative < 0.0

    def test_an_episode_inducing_no_error_is_not_scored(self) -> None:
        report = repair_scores(SETTLED, SETTLED, SETTLED, TARGET, REGION, LIVE)
        assert math.isnan(report.repaired_fraction)

    def test_padding_vertices_are_excluded(self) -> None:
        # Padding rows sum to 1 like any other; counting them would dilute both
        # halves towards zero and make G3 easier the more padding a mesh has.
        live = mask(True, True, False, False)
        report = repair_scores(SETTLED, DAMAGED, SETTLED, TARGET, REGION, live)
        assert report.repaired_fraction == 1.0
        assert report.collateral_relative == 0.0
