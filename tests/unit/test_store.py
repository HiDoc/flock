"""The .npz store round-trip and its schema guard."""

from __future__ import annotations

import numpy as np
import pytest

from flock.data.store import DatasetRepository
from flock.domain.errors import InvariantError


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/cache/d1").exists(), reason="D1 cache not built"
)
class TestRoundTrip:
    def test_a_cached_sample_reloads_and_revalidates(self) -> None:
        repo = DatasetRepository(__import__("pathlib").Path("data/cache/d1"))
        ids = repo.list_ids()
        assert ids, "cache is empty"
        sample, dense = repo.load(ids[0])
        assert sample.mesh.positions.shape == (1024, 3)
        assert sample.skeleton.num_bones == 22
        assert dense.shape == (1024, 22)
        # Reconstruction re-ran every value-object invariant; reaching here is the check.

    def test_split_is_reproducible(self) -> None:
        repo = DatasetRepository(__import__("pathlib").Path("data/cache/d1"))
        assert repo.split(seed=7).train == repo.split(seed=7).train
        assert repo.split(seed=7).train != repo.split(seed=8).train


def test_a_stale_schema_version_fails_loudly(tmp_path) -> None:
    """A cache from older code must fail at read time, not as a curve later."""
    repo = DatasetRepository(tmp_path)
    np.savez_compressed(repo.path_for(0), schema_version=999)
    with pytest.raises(InvariantError, match="rebuild the cache"):
        repo.load(0)
