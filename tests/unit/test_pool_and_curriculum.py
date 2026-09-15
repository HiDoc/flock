"""The state pool, the corruption curriculum and the run repository (M3)."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from flock.domain.dynamics.pool import CONVERGED_AGE, StatePool
from flock.domain.skinning.corruption import (
    CorruptionContext,
    CorruptionLevel,
    corrupt,
    grow_patch,
)
from flock.domain.skinning.weights import WeightField
from flock.experiment.repository import RunRepository, make_run_name
from flock.training.curriculum import (
    confident_error_curriculum,
    default_curriculum,
    sample_levels,
    stage_for,
    with_clean_share,
)

V, B = 1024, 8


def make_context(live: int = 200) -> CorruptionContext:
    vertex_mask = np.zeros(V, dtype=bool)
    vertex_mask[:live] = True
    neighbours = np.zeros((V, 8), dtype=np.int32)
    for i in range(live):
        neighbours[i] = (np.arange(8) + i + 1) % live
    return CorruptionContext(
        neighbours=neighbours,
        neighbour_mask=np.tile(vertex_mask[:, None], (1, 8)),
        vertex_mask=vertex_mask,
        candidate_bones=np.tile(np.arange(B, dtype=np.int32), (V, 1)),
        candidate_mask=np.ones((V, B), dtype=bool),
        candidate_distances=np.tile(np.linspace(0.1, 0.8, B).astype(np.float32), (V, 1)),
        bone_parents=np.array([-1, *range(B - 1)], dtype=np.int32),
    )


def make_field(live: int = 200) -> WeightField:
    values = np.zeros((V, B), dtype=np.float32)
    # Bone 2 dominates: mid-chain, so it has both a parent and a child.
    values[:live, 2] = 0.7
    values[:live, 3] = 0.3
    return WeightField(values)


class TestCorruption:
    def test_clean_leaves_the_field_untouched(self) -> None:
        field = make_field()
        out, mask = corrupt(field, CorruptionLevel.CLEAN, make_context(), np.random.default_rng(0))
        assert out is field
        assert not mask.any()

    @pytest.mark.parametrize(
        "level",
        [
            CorruptionLevel.C0_GAUSSIAN_LOGITS,
            CorruptionLevel.C1_LOCAL_PATCH,
            CorruptionLevel.C1P_PERMUTED_PATCH,
            CorruptionLevel.C2_HIERARCHY_TRANSFER,
            CorruptionLevel.C3_NAIVE_GEOMETRIC,
            CorruptionLevel.C4_NEAR_UNIFORM,
        ],
    )
    def test_every_level_returns_a_valid_field(self, level: CorruptionLevel) -> None:
        """Corruption must damage the answer, never the invariants."""
        out, _ = corrupt(make_field(), level, make_context(), np.random.default_rng(1))
        sums = out.values.sum(axis=1)
        live = sums > 0
        assert sums[live] == pytest.approx(1.0, abs=1e-4)
        assert (out.values >= 0).all()

    def test_c1_only_touches_its_patch(self) -> None:
        """Repair is scored inside the patch and collateral damage outside it,
        so the mask has to be honest."""
        field, context = make_field(), make_context()
        out, patch = corrupt(
            field, CorruptionLevel.C1_LOCAL_PATCH, context, np.random.default_rng(2)
        )
        untouched = context.vertex_mask & ~patch
        assert np.allclose(out.values[untouched], field.values[untouched])
        assert patch.sum() > 0

    def test_patch_is_connected_and_sized(self) -> None:
        context = make_context()
        patch = grow_patch(context, 0.1, np.random.default_rng(3))
        assert 1 <= patch.sum() <= 0.2 * context.vertex_mask.sum() + 2
        assert not patch[~context.vertex_mask].any()

    def test_c2_moves_mass_towards_the_hierarchy(self) -> None:
        field, context = make_field(), make_context()
        out, patch = corrupt(
            field, CorruptionLevel.C2_HIERARCHY_TRANSFER, context, np.random.default_rng(4)
        )
        rows = np.flatnonzero(patch)
        # Bone 2 dominated; its mass moves onto bone 1 or bone 3.
        assert out.values[rows, 2].mean() < field.values[rows, 2].mean()
        assert out.values[rows][:, [1, 3]].sum(axis=1).mean() > (
            field.values[rows][:, [1, 3]].sum(axis=1).mean()
        )

    def test_c1p_preserves_the_shape_of_every_row(self) -> None:
        """C1P is the peaked half of C1: the row keeps its distribution exactly
        and only its *identity* is wrong. That is the whole point — a corruption
        the model cannot spot by flatness alone (ADR-0008)."""
        field = make_field()
        out, patch = corrupt(
            field, CorruptionLevel.C1P_PERMUTED_PATCH, make_context(), np.random.default_rng(0)
        )
        rows = np.flatnonzero(patch)
        assert rows.size
        before = np.sort(field.values[rows], axis=1)
        after = np.sort(out.values[rows], axis=1)
        assert after == pytest.approx(before, abs=1e-6)

    def test_c1p_moves_the_dominant_bone(self) -> None:
        """Preserving the shape is only useful if the answer actually changes."""
        field = make_field()
        out, patch = corrupt(
            field, CorruptionLevel.C1P_PERMUTED_PATCH, make_context(), np.random.default_rng(0)
        )
        rows = np.flatnonzero(patch)
        moved = out.values[rows].argmax(axis=1) != field.values[rows].argmax(axis=1)
        assert moved.mean() > 0.5

    def test_c1p_keeps_mass_on_real_candidates(self) -> None:
        """A padding slot names no bone, so mass permuted onto one is nonsense."""
        context = make_context()
        mask = context.candidate_mask.copy()
        mask[:, 4:] = False
        narrowed = replace(context, candidate_mask=mask)
        out, patch = corrupt(
            make_field(), CorruptionLevel.C1P_PERMUTED_PATCH, narrowed,
            np.random.default_rng(0),
        )
        assert not out.values[np.flatnonzero(patch)][:, 4:].any()

    def test_c3_ignores_the_field_it_is_handed(self) -> None:
        """The R7 guarantee, as a test rather than a promise.

        C3 is the initialisation gate G4 starts from. If it read the weights it
        was passed, it would carry ground truth into the one measurement that
        exists to show the model can solve rather than repair — and the leak
        would be invisible in the result.
        """
        context = make_context()
        truth = make_field()
        scrambled = WeightField(np.zeros_like(truth.values))
        a, _ = corrupt(truth, CorruptionLevel.C3_NAIVE_GEOMETRIC, context,
                       np.random.default_rng(0))
        b, _ = corrupt(scrambled, CorruptionLevel.C3_NAIVE_GEOMETRIC, context,
                       np.random.default_rng(1))
        assert a.values == pytest.approx(b.values)

    def test_c3_favours_the_nearer_bone(self) -> None:
        """Inverse-square over candidate distance, so slot 0 must dominate."""
        out, region = corrupt(make_field(), CorruptionLevel.C3_NAIVE_GEOMETRIC,
                              make_context(), np.random.default_rng(0))
        rows = out.values[:200]
        assert (rows[:, 0] > rows[:, -1]).all()
        assert region[:200].all()          # a global initialisation, not a patch

    def test_c4_is_near_uniform(self) -> None:
        """C4 must leave the dynamics almost nothing to go on but its own rule."""
        out, _ = corrupt(make_field(), CorruptionLevel.C4_NEAR_UNIFORM,
                         make_context(), np.random.default_rng(0))
        rows = out.values[:200]
        assert rows.max(axis=1).mean() < 0.30       # uniform over 8 would be 0.125
        assert rows.min(axis=1).mean() > 0.01

    def test_c4_ignores_the_field_it_is_handed(self) -> None:
        context = make_context()
        a, _ = corrupt(make_field(), CorruptionLevel.C4_NEAR_UNIFORM, context,
                       np.random.default_rng(7))
        b, _ = corrupt(WeightField(np.zeros((V, B), dtype=np.float32)),
                       CorruptionLevel.C4_NEAR_UNIFORM, context, np.random.default_rng(7))
        assert a.values == pytest.approx(b.values)


class TestStatePool:
    def _pool(self) -> StatePool:
        return StatePool(num_meshes=4, vertices=V, hidden_dim=8, candidates=B, size=64, seed=0)

    def test_first_batch_resets_every_uninitialised_slot(self) -> None:
        """An unwritten slot cannot be rolled out; it must be reset first."""
        plan = self._pool().plan_batch(16)
        assert (plan.reset | plan.clean).all()

    def test_recorruption_waits_for_convergence(self) -> None:
        """Repair is meaningless applied to a state that never converged."""
        pool = self._pool()
        pool.initialised[:] = True
        pool.age[:] = CONVERGED_AGE - 1
        assert not self._plan_recorrupt(pool).any()
        pool.age[:] = CONVERGED_AGE
        # With every slot settled, some batch eventually draws into the band.
        assert any(self._plan_recorrupt(pool).any() for _ in range(40))

    @staticmethod
    def _plan_recorrupt(pool: StatePool) -> np.ndarray:
        return pool.plan_batch(16).recorrupt

    def test_the_three_fractions_never_overlap(self) -> None:
        pool = self._pool()
        pool.initialised[:] = True
        pool.age[:] = 32
        for _ in range(20):
            plan = pool.plan_batch(16)
            assert not (plan.reset & plan.clean).any()
            assert not (plan.recorrupt & plan.reset).any()
            assert not (plan.recorrupt & plan.clean).any()

    def test_write_advances_age_and_read_round_trips(self) -> None:
        pool = self._pool()
        slots = np.array([1, 5, 9])
        hidden = np.ones((3, V, 8), dtype=np.float32)
        logits = np.full((3, V, B), 0.5, dtype=np.float32)
        pool.write(slots, hidden, logits, steps=4)
        back_hidden, back_logits = pool.read(slots)
        assert np.allclose(back_hidden, hidden)
        assert np.allclose(back_logits, logits)
        assert (pool.age[slots] == 4).all()

    def test_reset_returns_age_to_zero(self) -> None:
        pool = self._pool()
        slots = np.array([2, 3])
        pool.write(slots, np.ones((2, V, 8), np.float32), np.ones((2, V, B), np.float32), 12)
        pool.reset_slots(
            slots, np.zeros((2, V, B), np.float32),
            [CorruptionLevel.C1_LOCAL_PATCH] * 2,
        )
        assert (pool.age[slots] == 0).all()
        assert np.allclose(pool.hidden[slots], 0.0)

    def test_age_histogram_reports_the_spread(self) -> None:
        """Ages stuck at zero mean the pool is churning; all-maximum means
        nothing is being reset. Both defeat its purpose."""
        pool = self._pool()
        assert pool.age_histogram()["age_max"] == 0.0
        pool.write(np.array([0, 1]), np.zeros((2, V, 8), np.float32),
                   np.zeros((2, V, B), np.float32), 8)
        assert pool.age_histogram()["age_max"] == 8.0


class TestCurriculum:
    def test_stages_advance_with_the_step(self) -> None:
        stages = default_curriculum()
        assert stage_for(stages, 0) is stages[0]
        assert stage_for(stages, 5_000) is stages[1]
        assert stage_for(stages, 50_000) is stages[2]

    def test_every_stage_keeps_clean_states(self) -> None:
        """The do-no-harm fraction is what stops collapse to the identity (R3)."""
        for stage in default_curriculum():
            assert stage.mixture.get(CorruptionLevel.CLEAN, 0.0) > 0.0

    def test_difficulty_increases(self) -> None:
        stages = default_curriculum()
        assert (
            stages[0].mixture[CorruptionLevel.C0_GAUSSIAN_LOGITS]
            > stages[2].mixture[CorruptionLevel.C0_GAUSSIAN_LOGITS]
        )

    def test_clean_share_moves_only_the_clean_to_corrupted_ratio(self) -> None:
        """The lever for ADR-0010: every other level's share must be untouched,
        or the arm tests two things at once."""
        base = confident_error_curriculum()
        lowered = with_clean_share(base, 0.1)
        for before, after in zip(base, lowered, strict=True):
            assert after.mixture[CorruptionLevel.CLEAN] == pytest.approx(0.1)
            assert sum(after.mixture.values()) == pytest.approx(sum(before.mixture.values()))
            for level in (CorruptionLevel.C1_LOCAL_PATCH, CorruptionLevel.C1P_PERMUTED_PATCH):
                assert after.mixture.get(level, 0.0) == pytest.approx(
                    before.mixture.get(level, 0.0)
                )

    def test_clean_share_refuses_an_impossible_value(self) -> None:
        with pytest.raises(ValueError, match="clean share"):
            with_clean_share(default_curriculum(), 1.0)

    def test_sampling_respects_the_mixture(self) -> None:
        drawn = sample_levels(default_curriculum(), 0, 4000, np.random.default_rng(0))
        share = sum(level is CorruptionLevel.CLEAN for level in drawn) / len(drawn)
        assert 0.25 < share < 0.35


class TestRunRepository:
    def test_create_record_and_load_round_trip(self, tmp_path) -> None:
        repo = RunRepository(tmp_path)
        name = make_run_name("v0", "round-trip", today=dt.date(2026, 8, 31))
        repo.create(name, {"lr": 0.001})
        repo.record_metrics(name, {"step": 1, "loss": 0.5})
        repo.record_metrics(name, {"step": 2, "loss": 0.4})
        record = repo.load(name)
        assert record.config["lr"] == 0.001
        assert [r["step"] for r in record.metrics["records"]] == [1, 2]
        assert repo.list_runs() == [name]

    def test_a_malformed_run_name_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="malformed run name"):
            RunRepository(tmp_path).create("not a run name", {})
