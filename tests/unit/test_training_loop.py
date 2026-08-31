"""The pooled training loop and by-iteration evaluation, end to end.

Skipped without the D1 cache: these exercise real data deliberately, because
the failure they guard against is a loop that runs happily on synthetic shapes
and produces nothing on real meshes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

CACHE = Path("data/cache/d1")
pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="D1 cache not built")


def test_a_short_run_writes_config_metrics_and_checkpoint(tmp_path) -> None:
    from flock.domain.dynamics.state import CellConfig
    from flock.training.loop import TrainConfig, train

    config = TrainConfig(
        cell=CellConfig(), batch_size=4, pool_size=16, max_steps=6, log_every=3,
        poses_per_step=2,
    )
    train(config, "v0_20260831_pytest-run", cache=CACHE, runs_root=tmp_path, num_meshes=4)

    run = tmp_path / "v0_20260831_pytest-run"
    assert (run / "config.json").exists()
    assert (run / "checkpoint.npz").exists()
    records = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [3, 6]
    for record in records:
        assert np.isfinite(record["loss"])
        assert record["l_weight"] >= 0.0


def test_pool_ages_advance_across_steps(tmp_path) -> None:
    """If ages stay at zero the pool is churning and no long horizon is seen."""
    from flock.domain.dynamics.state import CellConfig
    from flock.training.loop import TrainConfig, train

    config = TrainConfig(
        cell=CellConfig(), batch_size=4, pool_size=8, max_steps=20, log_every=10,
        poses_per_step=2,
    )
    train(config, "v0_20260831_pytest-age", cache=CACHE, runs_root=tmp_path, num_meshes=4)
    records = [
        json.loads(x)
        for x in (tmp_path / "v0_20260831_pytest-age" / "metrics.jsonl").read_text().splitlines()
    ]
    assert records[-1]["age_mean"] > records[0]["age_mean"] - 1e-9
    assert records[-1]["age_max"] >= 4


def test_evaluation_returns_a_curve_per_metric() -> None:
    """Every V0 result is a curve over T; nothing here may collapse to a scalar."""
    from flock.backends.mlx.cell import init_params
    from flock.data.store import DatasetRepository
    from flock.domain.dynamics.state import CellConfig
    from flock.evaluation.probe import evaluate_by_iteration

    repo = DatasetRepository(CACHE)
    samples = [repo.load(i)[0] for i in repo.list_ids()[:2]]
    config = CellConfig()
    initial = np.stack([s.ground_truth.values for s in samples])

    iterations = (0, 1, 2)
    curves = evaluate_by_iteration(
        init_params(config, 0), samples, config, initial, iterations=iterations
    )
    assert set(curves) == {"weight_l1", "deformation", "dice", "bleeding", "stability"}
    for curve in curves.values():
        assert len(curve.values) == len(iterations)

    # The untrained cell is the identity, so a clean input must stay put.
    assert curves["weight_l1"].values[0] == pytest.approx(0.0, abs=1e-4)
    assert curves["weight_l1"].values[-1] == pytest.approx(0.0, abs=1e-3)
