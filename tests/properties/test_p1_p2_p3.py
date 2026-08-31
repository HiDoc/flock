"""P1, P2, P3 — the falsifiable claims (spec §0.3).

The harnesses are written now and skipped until the numerics land, so that the
shape of the evidence is fixed before any result exists to be flattered by it.

* P1 — recurrence is useful: T iterations of one cell beat a single pass, and
  match or beat an unshared feed-forward GNN of equivalent depth (~8x params).
* P2 — there is an attractor: iterating past convergence does not degrade.
* P3 — it repairs: local corruption is resolved without collateral damage.

If P1 fails, the project is another GNN and must pivot or stop.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="numerics land at M3/M4")


def test_p1_recurrence_beats_single_pass() -> None:
    """L1(T=8) <= 0.8 * L1(T=1) on the probe set (gate G1)."""
    raise NotImplementedError("M4")


def test_p1_matches_unshared_feedforward_gnn() -> None:
    """Within 5% of baseline B3 at roughly an eighth of the parameters (G1)."""
    raise NotImplementedError("M4")


def test_p2_iterating_past_convergence_does_not_degrade() -> None:
    """Under 5% degradation from T=8 to T=32, stability decreasing (G2)."""
    raise NotImplementedError("M4")


def test_p3_local_corruption_is_repaired() -> None:
    """At least 80% of induced error resolved within 8 steps (G3)."""
    raise NotImplementedError("M4")


def test_p3_repair_does_no_collateral_damage() -> None:
    """Under 10% relative error increase outside the corrupted region (G3).

    The half of P3 that is easy to forget: a model that re-solves the entire
    mesh passes the repair half while failing the property.
    """
    raise NotImplementedError("M4")
