"""Seed replication reporting (spec §8)."""

from __future__ import annotations

import pytest

from flock.evaluation.replication import SEEDS_PER_CONCLUSION, Replicated


class TestReplicated:
    def test_mean_and_spread(self) -> None:
        r = Replicated("weight_l1@8", (0.030, 0.032, 0.034))
        assert r.mean == pytest.approx(0.032)
        assert r.spread == pytest.approx(0.002)

    def test_a_single_seed_has_no_spread_and_is_not_enough(self) -> None:
        r = Replicated("weight_l1@8", (0.030,))
        assert r.spread == 0.0
        assert not r.enough_seeds

    def test_three_seeds_meet_the_minimum(self) -> None:
        assert Replicated("m", (1.0, 2.0, 3.0)).enough_seeds
        assert SEEDS_PER_CONCLUSION == 3

    def test_straddling_seeds_are_flagged(self) -> None:
        """A mean that passes while its seeds disagree is not a conclusion."""
        assert Replicated("gain", (0.70, 0.75, 0.90)).straddles(0.80)
        assert not Replicated("gain", (0.70, 0.75, 0.78)).straddles(0.80)
        assert not Replicated("gain", (0.85, 0.90, 0.95)).straddles(0.80)

    def test_straddle_respects_the_direction_of_the_test(self) -> None:
        # A gate wanting values ABOVE a threshold reads the other way.
        assert Replicated("dice", (0.7, 0.95)).straddles(0.8, below_is_pass=False)
