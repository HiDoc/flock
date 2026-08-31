"""Run naming discipline (spec §8)."""

from __future__ import annotations

import datetime as dt

import pytest

from flock.experiment.repository import make_run_name


def test_builds_a_dated_run_name() -> None:
    name = make_run_name("v0", "delta-vs-absolute", today=dt.date(2026, 8, 31))
    assert name == "v0_20260831_delta-vs-absolute"


@pytest.mark.parametrize("change", ["Delta_Absolute", "two words", "UPPER", ""])
def test_rejects_names_that_break_the_convention(change: str) -> None:
    with pytest.raises(ValueError, match="malformed run name"):
        make_run_name("v0", change, today=dt.date(2026, 8, 31))


def test_accepts_every_project_version() -> None:
    for version in ("v0", "v0.5", "v1"):
        assert make_run_name(version, "baseline", today=dt.date(2026, 8, 31))
