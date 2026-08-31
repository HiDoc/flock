"""Offline preprocessing pipeline (spec §5, §2.2).

Runs once on CPU. Order matters and is fixed:

1. decimate to V=1024 (QEM)
2. normalise (unit height, centred)
3. build the one-ring, topped up by KNN to K=8
4. vertex / edge features
5. diffusion distances vertex-to-bone (heat method)
6. candidate selection, top-B by diffusion distance
7. pair features
8. reproject and renormalise GT weights onto the decimated mesh
9. pose banks by forward kinematics, train and test disjoint

Steps 1-7 never touch the ground truth; it enters only at step 8. That ordering
*is* the anti-leakage rule of spec R7 — expressed as control flow rather than as
a checklist item someone has to remember.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import igl
import numpy as np
from numpy.typing import NDArray

from flock.data.acl.articulation_xl import fold_to_canonical, to_skeleton
from flock.data.preprocess import decimate, diffusion, features, graph, poses
from flock.domain.geometry.mesh import NUM_CANDIDATES, NUM_NEIGHBOURS, NUM_VERTICES, Mesh
from flock.domain.geometry.pose import PoseBank
from flock.domain.sample import PreprocessedMesh
from flock.domain.skinning.weights import CandidateTable, WeightField


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing parameters (spec §5)."""

    target_vertices: int = NUM_VERTICES
    num_neighbours: int = NUM_NEIGHBOURS
    num_candidates: int = NUM_CANDIDATES
    train_poses: int = 24
    test_poses: int = 8
    joint_perturbation_degrees: float = 25.0
    """Range of the random articulation used to build pose banks."""

    diffusion_time: float = 1e-3
    """Heat-method time parameter; smaller approaches true geodesic distance."""

    candidate_metric: str = "euclidean"
    """Which distance ranks the candidates: `"euclidean"` or `"diffusion"`.

    H1 argues for diffusion, and gate G0 is what settles it. On D1 it does not
    survive contact: Euclidean selection scores 0.992/0.930 against diffusion's
    0.928/0.514, so diffusion selection would train under a ceiling it could
    never reach (R1). Diffusion distance is kept as a *pair feature* either way,
    so the anatomical signal H1 wants is still available to the cell — it is the
    ranking, not the information, that changes. Ablation A6 flips this.
    """


@dataclass(frozen=True)
class Preprocessed:
    """A finished sample plus the diagnostics G0 needs."""

    sample: PreprocessedMesh
    dense_weights: NDArray[np.float32]
    """`[V, J]` GT over every bone, kept so coverage can be scored honestly."""

    scale: float


def _pad(array: NDArray[Any], rows: int) -> NDArray[Any]:
    """Pad the leading axis up to `rows` with zeros."""
    if array.shape[0] >= rows:
        return array[:rows]
    pad = [(0, rows - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad)


def preprocess_entry(
    entry: dict[str, Any],
    mesh_id: int,
    config: PreprocessConfig,
    seed: int = 0,
) -> Preprocessed:
    """Run the full pipeline for one canonical corpus entry."""
    skeleton = to_skeleton(entry)
    dense_source = fold_to_canonical(entry)

    # 1-2. decimate, then normalise mesh and skeleton together.
    source_positions = np.asarray(entry["vertices"], dtype=np.float64)
    source_faces = np.asarray(entry["faces"], dtype=np.int32)
    small = decimate.decimate_to_vertex_budget(
        source_positions, source_faces, config.target_vertices
    )
    positions, scale = decimate.normalise(small.positions)
    centre = 0.5 * (small.positions.max(axis=0) + small.positions.min(axis=0))
    heads = (np.asarray(skeleton.heads, np.float64) - centre) / scale
    tails = (np.asarray(skeleton.tails, np.float64) - centre) / scale

    normals = np.asarray(igl.per_vertex_normals(positions, small.faces.astype(np.int64)))
    normals = np.nan_to_num(normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 1e-9, normals / np.maximum(lengths, 1e-12), [0.0, 0.0, 1.0])

    # 3-4. graph and geometry-only features.
    neighbours, neighbour_mask = graph.build_neighbours(
        positions, small.faces, config.num_neighbours
    )
    vertex_feat = features.vertex_features(positions, normals, neighbours, neighbour_mask)
    edge_feat = features.edge_features(positions, normals, neighbours)

    # 5-6. injected geodesy, then candidate selection.
    diffusion_distance = diffusion.vertex_to_bone_distances(
        positions, small.faces, heads, tails, config.diffusion_time
    )
    euclidean, abscissa = diffusion.euclidean_point_segment_distances(positions, heads, tails)

    live = len(positions)
    vertex_mask = np.zeros(NUM_VERTICES, dtype=bool)
    vertex_mask[:live] = True
    ranking = diffusion_distance if config.candidate_metric == "diffusion" else euclidean
    candidates = CandidateTable.from_diffusion_distance(_pad(ranking, NUM_VERTICES), vertex_mask)

    # 7. pair features.
    pair_feat = features.pair_features(
        _pad(normals, NUM_VERTICES),
        heads, tails, skeleton.depths(), candidates.bones,
        _pad(euclidean, NUM_VERTICES), _pad(abscissa, NUM_VERTICES),
        _pad(diffusion_distance, NUM_VERTICES),
    )

    # 8. ground truth enters only now.
    dense = _pad(decimate.transfer_weights(dense_source, small.source_vertex), NUM_VERTICES)
    rows = np.arange(NUM_VERTICES)[:, None]
    restricted = (dense[rows, candidates.bones] * candidates.mask).astype(np.float64)
    sums = restricted.sum(axis=1, keepdims=True)
    np.divide(restricted, sums, out=restricted, where=sums > 0)

    mesh = Mesh(
        _pad(positions, NUM_VERTICES).astype(np.float32),
        _pad(normals, NUM_VERTICES).astype(np.float32),
        small.faces.astype(np.int32),
        neighbours=_pad(neighbours, NUM_VERTICES),
        neighbour_mask=_pad(neighbour_mask, NUM_VERTICES),
        vertex_mask=vertex_mask,
    )

    # 9. disjoint pose banks.
    rng_train = np.random.default_rng(seed * 2 + 1)
    rng_test = np.random.default_rng(seed * 2 + 2)
    train = poses.sample_pose_bank(
        skeleton, config.train_poses, config.joint_perturbation_degrees, rng_train
    )
    test = poses.sample_pose_bank(
        skeleton, config.test_poses, config.joint_perturbation_degrees, rng_test
    )

    sample = PreprocessedMesh(
        mesh_id=mesh_id,
        source="d1",
        mesh=mesh,
        skeleton=skeleton,
        candidates=candidates,
        ground_truth=WeightField(restricted.astype(np.float32)),
        train_poses=PoseBank(train, "train"),
        test_poses=PoseBank(test, "test"),
        vertex_features=_pad(vertex_feat, NUM_VERTICES),
        edge_features=_pad(edge_feat, NUM_VERTICES),
        pair_features=pair_feat,
    )
    return Preprocessed(sample, dense.astype(np.float32), scale)
