"""Flow-character measures (spec §4.1; cf. arXiv 2604.12720)."""

from __future__ import annotations

import numpy as np
import pytest

from flock.evaluation.dynamics import flow_character, trajectories_converge

STEPS = 400


def test_a_straight_line_reads_as_one_dimensional() -> None:
    t = np.linspace(0, 1, STEPS)[:, None]
    line = t * np.array([[1.0, 2.0, -1.0, 0.5]])
    f = flow_character(line)
    assert f.cos_consecutive == pytest.approx(1.0, abs=1e-6)
    assert f.cos_endpoints == pytest.approx(1.0, abs=1e-6)
    assert f.dimension_95 == 1


def test_a_limit_cycle_needs_two_dimensions() -> None:
    """The case that must be distinguishable from drift: same failure on G2,
    opposite fix."""
    a = np.linspace(0, 8 * np.pi, STEPS)
    circle = np.stack([np.cos(a), np.sin(a), np.zeros_like(a), np.zeros_like(a)], axis=1)
    f = flow_character(circle)
    assert f.dimension_95 == 2
    assert f.pc1_share < 0.6
    assert f.cos_consecutive < 0.9995


def test_alternation_reads_as_negative() -> None:
    flip = np.zeros((STEPS, 2))
    flip[1::2, 0] = 1.0
    f = flow_character(flip)
    assert f.cos_consecutive < -0.9


def test_convergence_is_measured_at_the_endpoints() -> None:
    a = np.zeros((10, 16))
    b = np.linspace(1.0, 0.01, 10)[:, None] * np.ones((1, 16))
    converged, start, end = trajectories_converge(a, b, candidates=8)
    assert converged
    assert start > end

    apart = np.ones((10, 16))
    held, s0, s1 = trajectories_converge(a, apart, candidates=8)
    assert not held
    assert s0 == pytest.approx(s1)


def test_too_short_a_trajectory_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        flow_character(np.zeros((2, 4)))
