"""The cell, the rollout and the fused LBS (M2).

Synthetic inputs throughout, so the suite runs without the corpus or the cache.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from flock.backends.mlx.cell import cell_step, gru_cell, init_params, weights_of
from flock.backends.mlx.lbs import lbs
from flock.backends.mlx.rollout import rollout, rollout_trace
from flock.data.preprocess.features import (
    EDGE_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    VERTEX_FEATURE_DIM,
)
from flock.domain.dynamics.state import CellConfig, CellState, StaticInputs

N, V, K, B, H = 2, 32, 8, 8, 32


def make_static(live: int = V) -> StaticInputs:
    rng = np.random.default_rng(0)
    vertex_mask = np.zeros((N, V), dtype=np.float32)
    vertex_mask[:, :live] = 1.0
    neighbours = rng.integers(0, live, (N, V, K)).astype(np.int32)
    return StaticInputs(
        vertex_features=mx.array(rng.normal(size=(N, V, VERTEX_FEATURE_DIM)).astype(np.float32)),
        edge_features=mx.array(rng.normal(size=(N, V, K, EDGE_FEATURE_DIM)).astype(np.float32)),
        pair_features=mx.array(rng.normal(size=(N, V, B, PAIR_FEATURE_DIM)).astype(np.float32)),
        neighbours=mx.array(neighbours),
        neighbour_mask=mx.array(np.ones((N, V, K), dtype=np.float32)),
        candidate_mask=mx.array(np.ones((N, V, B), dtype=np.float32)),
        vertex_mask=mx.array(vertex_mask),
    )


def make_state() -> CellState:
    rng = np.random.default_rng(1)
    return CellState(
        hidden=mx.zeros((N, V, H)),
        logits=mx.array(rng.normal(size=(N, V, B)).astype(np.float32)),
    )


class TestCell:
    def test_parameter_count_is_far_under_the_brief(self) -> None:
        """H6: the brief's 1-2M was an upper bound, not a target."""
        params = init_params(CellConfig(), 0)
        total = sum(v.size for group in params.values() for v in group.values())
        assert 20_000 < total < 100_000

    def test_untrained_cell_is_the_identity_on_logits(self) -> None:
        """MLP_out starts at zero, so dz = 0 and the first step changes nothing.

        This is the do-no-harm prior (R3) and the stable starting point for P2.
        """
        config, static, state = CellConfig(), make_static(), make_state()
        params = init_params(config, 0)
        stepped = cell_step(params, state, static, config)
        assert np.allclose(np.array(stepped.logits), np.array(state.logits), atol=1e-6)

    def test_the_hidden_state_does_move(self) -> None:
        """The gate must not be inert, or nothing could ever be learned."""
        config, static, state = CellConfig(), make_static(), make_state()
        stepped = cell_step(init_params(config, 0), state, static, config)
        assert np.abs(np.array(stepped.hidden)).max() > 0

    def test_delta_scale_damps_the_update(self) -> None:
        """alpha scales the logit update linearly (A4's knob)."""
        static, state = make_static(), make_state()
        params = init_params(CellConfig(), 0)
        params["out"]["w2"] = mx.ones(params["out"]["w2"].shape) * 0.01
        small = cell_step(params, state, static, CellConfig(delta_scale=0.1))
        large = cell_step(params, state, static, CellConfig(delta_scale=0.2))
        d_small = np.array(small.logits - state.logits)
        d_large = np.array(large.logits - state.logits)
        assert np.allclose(2 * d_small, d_large, atol=1e-5)

    def test_padded_vertices_are_never_written(self) -> None:
        config, state = CellConfig(), make_state()
        static = make_static(live=16)
        stepped = cell_step(init_params(config, 0), state, static, config)
        dead = np.array(static.vertex_mask)[0] == 0
        assert np.allclose(np.array(stepped.hidden)[0][dead], 0.0)

    def test_weights_form_a_partition_of_unity(self) -> None:
        config, static = CellConfig(), make_static()
        weights = np.array(weights_of(make_state(), config, static))
        assert weights.sum(axis=-1) == pytest.approx(1.0, abs=1e-5)
        assert (weights >= 0).all()

    def test_masked_candidates_receive_no_weight(self) -> None:
        config, state = CellConfig(), make_state()
        static = make_static()
        mask = np.array(static.candidate_mask)
        mask[:, :, 4:] = 0.0
        static = StaticInputs(**{**static.as_tree(), "candidate_mask": mx.array(mask)})
        weights = np.array(weights_of(state, config, static))
        assert weights[:, :, 4:] == pytest.approx(0.0, abs=1e-6)
        assert weights[:, :, :4].sum(axis=-1) == pytest.approx(1.0, abs=1e-5)


class TestGru:
    def test_a_saturated_gate_holds_the_state(self) -> None:
        """The learned way to do nothing — the quiescence Reynolds lacks."""
        hidden = mx.array(np.random.default_rng(0).normal(size=(3, H)).astype(np.float32))
        inputs = mx.zeros((3, 12))
        params = {
            # Bias alone drives the gate, so the test does not depend on inputs.
            "wz": mx.zeros((12 + H, H)), "bz": mx.ones((H,)) * 50.0,
            "wr": mx.zeros((12 + H, H)), "br": mx.zeros((H,)),
            "wn": mx.zeros((12 + H, H)), "bn": mx.zeros((H,)),
        }
        assert np.allclose(np.array(gru_cell(params, hidden, inputs)), np.array(hidden), atol=1e-4)


class TestRollout:
    def test_shapes_survive_any_number_of_steps(self) -> None:
        config, static, state = CellConfig(), make_static(), make_state()
        params = init_params(config, 0)
        for steps in (0, 1, 4, 9):
            out = rollout(params, state, static, steps, config)
            assert out.logits.shape == state.logits.shape
            assert out.age == steps

    def test_trace_captures_each_requested_iteration(self) -> None:
        config, static, state = CellConfig(), make_static(), make_state()
        captured = rollout_trace(init_params(config, 0), state, static, (0, 1, 2, 4), config)
        assert sorted(captured) == [0, 1, 2, 4]
        assert captured[0] is state


class TestFusedLbs:
    def test_identity_transform_leaves_vertices_at_rest(self) -> None:
        rng = np.random.default_rng(0)
        positions = mx.array(rng.normal(size=(1, V, 3)).astype(np.float32))
        weights = np.abs(rng.normal(size=(1, V, B))).astype(np.float32)
        weights /= weights.sum(-1, keepdims=True)
        identity = np.zeros((1, 3, 22, 3, 4), dtype=np.float32)
        identity[..., :3] = np.eye(3)
        bones = mx.array(rng.integers(0, 22, (1, V, B)).astype(np.int32))
        posed = np.array(lbs(positions, mx.array(weights), bones, mx.array(identity)))
        assert posed[0, 0] == pytest.approx(np.array(positions)[0], abs=1e-5)
