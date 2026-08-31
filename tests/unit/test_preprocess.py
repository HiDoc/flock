"""Preprocessing: welding, decimation, kinematics and the store round-trip.

Built on synthetic geometry so the suite runs without the 37 GB corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from flock.data.preprocess import decimate, diffusion, poses
from flock.domain.geometry.skeleton import ROOT_PARENT, Skeleton


def split_seam_grid(side: int = 12):
    """A grid emitted as disjoint triangles, the way an FBX export arrives."""
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    grid = np.stack([xs, ys, np.zeros_like(xs)], axis=-1).reshape(-1, 3).astype(np.float64)
    tris = []
    for i in range(side - 1):
        for j in range(side - 1):
            a, b = i * side + j, i * side + j + 1
            c, d = (i + 1) * side + j, (i + 1) * side + j + 1
            tris += [[a, b, c], [b, d, c]]
    faces = np.array(tris, dtype=np.int32)
    # Explode: every triangle gets its own copy of each vertex.
    exploded = grid[faces.reshape(-1)]
    return exploded, np.arange(len(exploded), dtype=np.int32).reshape(-1, 3)


class TestWeld:
    def test_welding_reunites_split_seams(self) -> None:
        positions, faces = split_seam_grid()
        welded, welded_faces, birth = decimate.weld(positions, faces)
        assert len(welded) < len(positions)
        assert len(np.unique(np.round(welded, 6), axis=0)) == len(welded)
        assert welded_faces.max() < len(welded)
        assert birth.max() < len(positions)

    def test_welding_collapses_the_component_count(self) -> None:
        """The reason welding is correctness, not tidiness: geodesy needs
        connectivity, and split seams destroy it."""
        positions, faces = split_seam_grid()
        before, _ = diffusion.connected_components(len(positions), faces)
        welded, welded_faces, _ = decimate.weld(positions, faces)
        after, _ = diffusion.connected_components(len(welded), welded_faces)
        assert before > 100
        assert after == 1


class TestDecimate:
    @pytest.mark.parametrize("budget", [32, 64, 200])
    def test_budget_is_never_exceeded(self, budget: int) -> None:
        positions, faces = split_seam_grid(20)
        result = decimate.decimate_to_vertex_budget(positions, faces, budget)
        assert len(result.positions) <= budget

    def test_no_vertex_is_left_without_a_face(self) -> None:
        """geometry-central segfaults on an unreferenced vertex; it does not raise."""
        positions, faces = split_seam_grid(20)
        result = decimate.decimate_to_vertex_budget(positions, faces, 64)
        assert len(np.unique(result.faces)) == len(result.positions)

    def test_weights_stay_a_partition_of_unity(self) -> None:
        positions, faces = split_seam_grid(20)
        result = decimate.decimate_to_vertex_budget(positions, faces, 64)
        weights = np.random.default_rng(0).random((len(positions), 5)).astype(np.float32)
        weights /= weights.sum(axis=1, keepdims=True)
        moved = decimate.transfer_weights(weights, result.source_vertex)
        assert moved.sum(axis=1) == pytest.approx(1.0, abs=1e-5)

    def test_normalise_centres_and_scales(self) -> None:
        positions, _ = split_seam_grid()
        centred, scale = decimate.normalise(positions)
        assert np.abs(centred).max() == pytest.approx(0.5, abs=1e-6)
        assert scale > 0


class TestEuclideanDistance:
    def test_distance_and_abscissa_on_a_known_segment(self) -> None:
        points = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 0.0], [2.0, 0.0, 0.0]])
        heads = np.array([[0.0, 0.0, 0.0]])
        tails = np.array([[1.0, 0.0, 0.0]])
        distance, t = diffusion.euclidean_point_segment_distances(points, heads, tails)
        assert distance[:, 0] == pytest.approx([1.0, 2.0, 1.0])
        assert t[:, 0] == pytest.approx([0.0, 0.5, 1.0])  # clamped past the end

    def test_zero_length_bone_collapses_to_its_point(self) -> None:
        """The root bone is degenerate by construction; distance to it is
        distance to the joint."""
        points = np.array([[3.0, 4.0, 0.0]])
        origin = np.zeros((1, 3))
        distance, t = diffusion.euclidean_point_segment_distances(points, origin, origin)
        assert distance[0, 0] == pytest.approx(5.0)
        assert t[0, 0] == 0.0


def chain_skeleton(num: int = 4) -> Skeleton:
    positions = np.zeros((num, 3), dtype=np.float32)
    positions[:, 1] = np.arange(num)
    heads = np.vstack([positions[0], positions[:-1]])
    parents = np.array([ROOT_PARENT, *range(num - 1)], dtype=np.int32)
    return Skeleton(heads, positions, parents, tuple(f"j{i}" for i in range(num)))


class TestPoseBank:
    def test_transforms_are_rigid(self) -> None:
        bank = poses.sample_pose_bank(chain_skeleton(), 6, 25.0, np.random.default_rng(0))
        rotations = bank[:, :, :, :3]
        identity = np.einsum("pjab,pjcb->pjac", rotations, rotations)
        assert identity == pytest.approx(np.broadcast_to(np.eye(3), identity.shape), abs=1e-5)
        assert np.linalg.det(rotations) == pytest.approx(1.0, abs=1e-5)

    def test_zero_perturbation_is_the_identity_transform(self) -> None:
        """With no rotation the rig must not move a single vertex."""
        skeleton = chain_skeleton()
        bank = poses.sample_pose_bank(skeleton, 2, 0.0, np.random.default_rng(0))
        point = np.array([0.3, 1.7, -0.2])
        for pose in bank:
            for bone in range(skeleton.num_bones):
                moved = pose[bone, :, :3] @ point + pose[bone, :, 3]
                assert moved == pytest.approx(point, abs=1e-5)

    def test_rotation_propagates_down_the_hierarchy(self) -> None:
        skeleton = chain_skeleton(5)
        bank = poses.sample_pose_bank(skeleton, 1, 25.0, np.random.default_rng(3))
        rest = np.asarray(skeleton.tails, dtype=np.float64)
        posed = np.einsum("jab,jb->ja", bank[0, :, :, :3], rest) + bank[0, :, :, 3]
        # A child joint must be displaced at least as much as its parent.
        drift = np.linalg.norm(posed - rest, axis=1)
        assert drift[-1] >= drift[1] - 1e-6
