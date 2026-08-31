"""Value-object invariants (spec R1, R7).

These are the tests that matter most at M0. The two silent-failure risks in the
spec — a candidate set that cannot reach the ground truth, and ground truth
leaking into features — are data problems, and a data problem that is not caught
at construction is caught three days later as a curve nobody can explain.
"""

from __future__ import annotations

import numpy as np
import pytest

from flock.domain.errors import InvariantError
from flock.domain.geometry.mesh import NUM_CANDIDATES, NUM_VERTICES, Mesh
from flock.domain.geometry.pose import PoseBank
from flock.domain.geometry.skeleton import ROOT_PARENT, Skeleton
from flock.domain.skinning.weights import CandidateTable, WeightField

V, B = NUM_VERTICES, NUM_CANDIDATES


def make_mesh(real: int = 64) -> Mesh:
    positions = np.zeros((V, 3), dtype=np.float32)
    positions[:real] = np.linspace(0, 1, real * 3, dtype=np.float32).reshape(real, 3)
    normals = np.zeros((V, 3), dtype=np.float32)
    normals[:, 0] = 1.0
    vertex_mask = np.zeros(V, dtype=bool)
    vertex_mask[:real] = True
    neighbours = np.zeros((V, 8), dtype=np.int32)
    neighbour_mask = np.zeros((V, 8), dtype=bool)
    for i in range(real):
        neighbours[i] = (np.arange(8) + i + 1) % real
        neighbour_mask[i] = True
    faces = np.stack([np.arange(real - 2), np.arange(1, real - 1), np.arange(2, real)], axis=1)
    return Mesh(positions, normals, faces.astype(np.int32), neighbours, neighbour_mask, vertex_mask)


def make_skeleton(j: int = 4) -> Skeleton:
    heads = np.zeros((j, 3), dtype=np.float32)
    tails = np.ones((j, 3), dtype=np.float32)
    parents = np.array([ROOT_PARENT, *range(j - 1)], dtype=np.int32)
    return Skeleton(heads, tails, parents, tuple(f"bone{i}" for i in range(j)))


class TestMesh:
    def test_valid_mesh_constructs(self) -> None:
        assert make_mesh().num_real_vertices == 64

    def test_rejects_wrong_vertex_count(self) -> None:
        mesh = make_mesh()
        with pytest.raises(InvariantError, match="positions must be"):
            Mesh(
                mesh.positions[:512], mesh.normals, mesh.faces,
                mesh.neighbours, mesh.neighbour_mask, mesh.vertex_mask,
            )

    def test_rejects_neighbour_pointing_at_padding(self) -> None:
        mesh = make_mesh()
        neighbours = mesh.neighbours.copy()
        neighbours[0, 0] = 900  # padded region
        with pytest.raises(InvariantError, match="padded vertex"):
            Mesh(
                mesh.positions, mesh.normals, mesh.faces,
                neighbours, mesh.neighbour_mask, mesh.vertex_mask,
            )

    def test_rejects_non_unit_normals(self) -> None:
        mesh = make_mesh()
        with pytest.raises(InvariantError, match="unit length"):
            Mesh(
                mesh.positions, mesh.normals * 2.0, mesh.faces,
                mesh.neighbours, mesh.neighbour_mask, mesh.vertex_mask,
            )


class TestSkeleton:
    def test_valid_skeleton_constructs(self) -> None:
        assert make_skeleton().num_bones == 4

    def test_rejects_two_roots(self) -> None:
        with pytest.raises(InvariantError, match="exactly one root"):
            Skeleton(
                np.zeros((3, 3), np.float32), np.ones((3, 3), np.float32),
                np.array([ROOT_PARENT, ROOT_PARENT, 0], np.int32), ("a", "b", "c"),
            )

    def test_rejects_parent_cycle(self) -> None:
        with pytest.raises(InvariantError, match="cycle"):
            Skeleton(
                np.zeros((3, 3), np.float32), np.ones((3, 3), np.float32),
                np.array([ROOT_PARENT, 2, 1], np.int32), ("a", "b", "c"),
            )

    def test_depths_follow_the_hierarchy(self) -> None:
        assert make_skeleton().depths().tolist() == [0, 1, 2, 3]


class TestWeightField:
    def test_rows_summing_to_one_are_accepted(self) -> None:
        values = np.zeros((V, B), dtype=np.float32)
        values[:100] = 1.0 / B
        assert WeightField(values).values.shape == (V, B)

    def test_rejects_rows_that_do_not_sum_to_one(self) -> None:
        values = np.zeros((V, B), dtype=np.float32)
        values[0] = 0.5 / B
        with pytest.raises(InvariantError, match="sum to 1"):
            WeightField(values)

    def test_rejects_negative_weights(self) -> None:
        values = np.zeros((V, B), dtype=np.float32)
        values[0, 0], values[0, 1] = 1.5, -0.5
        with pytest.raises(InvariantError, match="negative"):
            WeightField(values)

    def test_support_uses_an_explicit_threshold(self) -> None:
        values = np.zeros((V, B), dtype=np.float32)
        values[0] = np.array([0.97, 0.03] + [0.0] * (B - 2), dtype=np.float32)
        assert WeightField(values).support(threshold=0.05)[0].sum() == 1
        assert WeightField(values).support(threshold=0.01)[0].sum() == 2


class TestCandidateTable:
    def _table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bones = np.tile(np.arange(B, dtype=np.int32), (V, 1))
        mask = np.ones((V, B), dtype=bool)
        distances = np.tile(np.arange(B, dtype=np.float32), (V, 1))
        return bones, mask, distances

    def test_valid_table_constructs(self) -> None:
        assert CandidateTable(*self._table()).bones.shape == (V, B)

    def test_rejects_duplicate_bone_in_a_row(self) -> None:
        bones, mask, distances = self._table()
        bones[0, 1] = bones[0, 0]
        with pytest.raises(InvariantError, match="same candidate bone twice"):
            CandidateTable(bones, mask, distances)

    def test_rejects_negative_distance(self) -> None:
        bones, mask, distances = self._table()
        distances[0, 0] = -1.0
        with pytest.raises(InvariantError, match="distance is negative"):
            CandidateTable(bones, mask, distances)

    def test_selection_entry_point_cannot_receive_ground_truth(self) -> None:
        """Risk R7, made structural: the only supported constructor for a
        candidate table takes geometry and nothing else."""
        import inspect

        params = set(inspect.signature(CandidateTable.from_diffusion_distance).parameters)
        assert params == {"distances", "vertex_mask"}
        assert not any("weight" in p or "truth" in p or "gt" in p for p in params)


class TestPoseBank:
    def test_valid_bank_constructs(self) -> None:
        bank = PoseBank(np.zeros((8, 4, 3, 4), dtype=np.float32), "test")
        assert bank.num_poses == 8
        assert bank.num_bones == 4

    def test_rejects_unknown_split(self) -> None:
        with pytest.raises(InvariantError, match="split must be"):
            PoseBank(np.zeros((8, 4, 3, 4), dtype=np.float32), "validation")

    def test_rejects_empty_bank(self) -> None:
        with pytest.raises(InvariantError, match="empty"):
            PoseBank(np.zeros((0, 4, 3, 4), dtype=np.float32), "train")
