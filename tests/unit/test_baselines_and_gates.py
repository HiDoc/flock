"""Baselines, ablation switches and gate arithmetic (M4)."""

from __future__ import annotations

import numpy as np
import pytest

from flock.backends.mlx.cell import cell_step, init_params, layer_params
from flock.domain.dynamics.state import CellConfig
from flock.evaluation.baselines import ReynoldsWeights, reynolds_step
from flock.evaluation.gates import GateStatus, evaluate_gate


class TestUnsharedWeights:
    def test_unshared_multiplies_the_parameter_count(self) -> None:
        """B3 is the same depth at T times the parameters (spec §4.2)."""
        shared = init_params(CellConfig(shared_weights=True), 0)
        unshared = init_params(CellConfig(shared_weights=False), 0, depth=5)
        n_shared = sum(v.size for g in shared.values() for v in g.values())
        n_unshared = sum(v.size for g in unshared.values() for v in g.values())
        assert n_unshared == 5 * n_shared

    def test_each_step_draws_different_parameters(self) -> None:
        unshared = init_params(CellConfig(shared_weights=False), 0, depth=4)
        first = layer_params(unshared, 0, shared=False)["msg"]["w1"]
        second = layer_params(unshared, 1, shared=False)["msg"]["w1"]
        assert not np.allclose(np.array(first), np.array(second))

    def test_indexing_cycles_past_the_trained_depth(self) -> None:
        """A T-sweep must run; past its depth an unshared model repeats."""
        unshared = init_params(CellConfig(shared_weights=False), 0, depth=3)
        assert np.allclose(
            np.array(layer_params(unshared, 0, shared=False)["msg"]["w1"]),
            np.array(layer_params(unshared, 3, shared=False)["msg"]["w1"]),
        )

    def test_shared_mode_ignores_the_step_index(self) -> None:
        shared = init_params(CellConfig(shared_weights=True), 0)
        assert layer_params(shared, 7, shared=True) is shared


class TestAbsoluteLogits:
    def _run(self, absolute: bool):
        from tests.unit.test_cell import make_state, make_static

        config = CellConfig(absolute_logits=absolute)
        params = init_params(config, 0)
        params["out"]["w2"] = np.float32(0.05) * np.ones(params["out"]["w2"].shape)
        import mlx.core as mx

        params["out"]["w2"] = mx.array(np.array(params["out"]["w2"]))
        state, static = make_state(), make_static()
        return state, cell_step(params, state, static, config)

    def test_delta_mode_accumulates_on_the_previous_logits(self) -> None:
        state, stepped = self._run(absolute=False)
        assert np.abs(np.array(stepped.logits - state.logits)).max() < 1.0

    def test_absolute_mode_discards_the_previous_logits(self) -> None:
        """A4: predicting outright makes the last iteration overwrite the rest,
        so no fixed point is representable."""
        state, stepped = self._run(absolute=True)
        assert not np.allclose(np.array(stepped.logits), np.array(state.logits))


class TestReynolds:
    def _sample(self):
        class Fake:
            pass

        num_v, num_b, num_k = 40, 8, 4
        s, mesh, cand = Fake(), Fake(), Fake()
        mesh.neighbours = np.tile(np.arange(num_k, dtype=np.int32), (num_v, 1))
        mesh.neighbour_mask = np.ones((num_v, num_k), dtype=bool)
        mesh.vertex_mask = np.ones(num_v, dtype=bool)
        cand.distances = np.tile(np.linspace(0.1, 1.0, num_b).astype(np.float32), (num_v, 1))
        cand.mask = np.ones((num_v, num_b), dtype=bool)
        s.mesh, s.candidates = mesh, cand
        return s

    def test_a_step_preserves_the_partition_of_unity(self) -> None:
        sample = self._sample()
        weights = np.full((40, 8), 1.0 / 8, dtype=np.float32)
        out = reynolds_step(weights, sample, ReynoldsWeights(0.5, 0.5, 0.5, 0.3))
        assert out.sum(axis=1) == pytest.approx(1.0, abs=1e-5)
        assert (out >= 0).all()

    def test_zero_coefficients_leave_the_field_alone(self) -> None:
        """The identity has to be reachable, or the grid cannot find it."""
        sample = self._sample()
        weights = np.abs(np.random.default_rng(0).random((40, 8))).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        out = reynolds_step(weights, sample, ReynoldsWeights(0.0, 0.0, 0.0, 0.5))
        assert out == pytest.approx(weights, abs=1e-5)


class TestGates:
    def test_a_missing_metric_is_not_run_never_a_pass(self) -> None:
        assert evaluate_gate("G1", {"l1_t1": 0.1}).status is GateStatus.NOT_RUN

    def test_g1_needs_both_halves(self) -> None:
        base = {"l1_t1": 1.0, "l1_t8": 0.5, "params": 33_665, "b3_params": 168_325}
        # Recurrence gain alone is not enough if B3 is clearly better.
        assert evaluate_gate("G1", {**base, "b3_l1_t8": 0.2}).status is GateStatus.FAIL
        assert evaluate_gate("G1", {**base, "b3_l1_t8": 0.5}).status is GateStatus.PASS

    def test_g1_requires_the_model_to_be_smaller(self) -> None:
        metrics = {"l1_t1": 1.0, "l1_t8": 0.5, "b3_l1_t8": 0.5,
                   "params": 200_000, "b3_params": 168_325}
        assert evaluate_gate("G1", metrics).status is GateStatus.FAIL

    def test_g2_allows_five_percent_drift(self) -> None:
        assert evaluate_gate("G2", {"l1_t8": 0.10, "l1_t32": 0.104}).status is GateStatus.PASS
        assert evaluate_gate("G2", {"l1_t8": 0.10, "l1_t32": 0.120}).status is GateStatus.FAIL

    def test_g3_scores_repair_and_collateral_damage_together(self) -> None:
        """Re-solving the whole mesh passes the repair half and fails P3."""
        good = {"repaired_fraction": 0.9, "collateral_relative": 0.05}
        assert evaluate_gate("G3", good).status is GateStatus.PASS
        assert evaluate_gate(
            "G3", {**good, "collateral_relative": 0.5}
        ).status is GateStatus.FAIL
