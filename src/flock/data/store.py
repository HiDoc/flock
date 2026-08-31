"""Dataset repository — the `.npz` store (spec §5, §3.5, ADR-0003).

Fixed-shape `.npz` is the framework-neutral seam of the whole system: it is what
makes a second backend a sibling directory rather than a rewrite (ADR-0002).

`load` always runs `PreprocessedMesh.validate()`. A cache written by older code
must fail at read time, not surface later as a curve nobody can reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flock.domain.errors import InvariantError
from flock.domain.geometry.mesh import Mesh
from flock.domain.geometry.pose import PoseBank
from flock.domain.geometry.skeleton import Skeleton
from flock.domain.sample import SCHEMA_VERSION, PreprocessedMesh
from flock.domain.skinning.weights import CandidateTable, WeightField


@dataclass(frozen=True)
class DatasetSplit:
    """Train/validation split plus a held-out category (spec §5).

    The held-out category gives a weak generalisation signal. Weak is the honest
    word for it: V0 is asking about the dynamics, not about generalisation.
    """

    train: tuple[int, ...]
    val: tuple[int, ...]
    held_out: tuple[int, ...]


class DatasetRepository:
    """Reads and writes preprocessed meshes under a cache root."""

    def __init__(self, root: Path) -> None:
        """Bind the repository to a cache directory."""
        self.root = root

    def path_for(self, mesh_id: int) -> Path:
        """Location of one sample on disk."""
        return self.root / f"mesh_{mesh_id:06d}.npz"

    def save(self, sample: PreprocessedMesh, dense_weights: np.ndarray) -> Path:
        """Serialise one sample, plus the dense GT the coverage oracle needs."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(sample.mesh_id)
        np.savez_compressed(
            path,
            schema_version=SCHEMA_VERSION,
            mesh_id=sample.mesh_id,
            source=sample.source,
            positions=sample.mesh.positions,
            normals=sample.mesh.normals,
            faces=sample.mesh.faces,
            neighbours=sample.mesh.neighbours,
            neighbour_mask=sample.mesh.neighbour_mask,
            vertex_mask=sample.mesh.vertex_mask,
            heads=sample.skeleton.heads,
            tails=sample.skeleton.tails,
            parents=sample.skeleton.parents,
            names=np.array(sample.skeleton.names),
            candidate_bones=sample.candidates.bones,
            candidate_mask=sample.candidates.mask,
            candidate_distances=sample.candidates.distances,
            ground_truth=sample.ground_truth.values,
            dense_weights=dense_weights,
            train_poses=sample.train_poses.transforms,
            test_poses=sample.test_poses.transforms,
            vertex_features=sample.vertex_features,
            edge_features=sample.edge_features,
            pair_features=sample.pair_features,
        )
        return path

    def load(self, mesh_id: int) -> tuple[PreprocessedMesh, np.ndarray]:
        """Read one sample, validating the schema version and every invariant."""
        with np.load(self.path_for(mesh_id), allow_pickle=False) as data:
            version = int(data["schema_version"])
            if version != SCHEMA_VERSION:
                raise InvariantError(
                    f"cache schema {version} != expected {SCHEMA_VERSION}; rebuild the cache"
                )
            sample = PreprocessedMesh(
                mesh_id=int(data["mesh_id"]),
                source=str(data["source"]),
                mesh=Mesh(
                    data["positions"], data["normals"], data["faces"],
                    data["neighbours"], data["neighbour_mask"], data["vertex_mask"],
                ),
                skeleton=Skeleton(
                    data["heads"], data["tails"], data["parents"],
                    tuple(str(n) for n in data["names"]),
                ),
                candidates=CandidateTable(
                    data["candidate_bones"], data["candidate_mask"], data["candidate_distances"],
                ),
                ground_truth=WeightField(data["ground_truth"]),
                train_poses=PoseBank(data["train_poses"], "train"),
                test_poses=PoseBank(data["test_poses"], "test"),
                vertex_features=data["vertex_features"],
                edge_features=data["edge_features"],
                pair_features=data["pair_features"],
            )
            return sample, data["dense_weights"]

    def list_ids(self) -> list[int]:
        """Cached mesh ids, in order."""
        return sorted(int(p.stem.split("_")[1]) for p in self.root.glob("mesh_*.npz"))

    def split(self, seed: int, val_fraction: float = 0.2) -> DatasetSplit:
        """Build a reproducible split over the cached ids."""
        ids = np.array(self.list_ids())
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        cut = int(len(ids) * (1.0 - val_fraction))
        return DatasetSplit(tuple(int(i) for i in ids[:cut]), tuple(int(i) for i in ids[cut:]), ())
