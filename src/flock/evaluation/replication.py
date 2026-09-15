"""Seed replication (spec §8).

"Three seeds per config for any conclusion" is a rule that decays into a single
lucky run under time pressure, so it lives here as code rather than as a habit.

Nothing in this module reports a mean without also reporting the spread and the
individual values. A gate decided on a mean whose seeds straddle the threshold
has not been decided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SEEDS_PER_CONCLUSION = 3


@dataclass(frozen=True)
class Replicated:
    """One metric measured across seeds."""

    metric: str
    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        """Mean across seeds."""
        return float(np.mean(self.values))

    @property
    def spread(self) -> float:
        """Sample standard deviation; 0.0 for a single seed."""
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 0.0

    @property
    def enough_seeds(self) -> bool:
        """Whether §8's three-seed minimum is met."""
        return len(self.values) >= SEEDS_PER_CONCLUSION

    def straddles(self, threshold: float, below_is_pass: bool = True) -> bool:
        """True if the seeds fall on both sides of `threshold`.

        A conclusion whose seeds disagree is not a conclusion, however the mean
        happens to land.
        """
        passes = [(v <= threshold) == below_is_pass for v in self.values]
        return any(passes) and not all(passes)


def load_checkpoint(run_dir: Path, config: Any, backend_name: str = "mlx") -> Any:
    """Read a parameter tree written by `RunRepository.save_checkpoint`."""
    from flock.backends.registry import get_backend

    return get_backend(backend_name, config).load_params(run_dir / "checkpoint.npz")


def replicate(
    run_dirs: list[Path],
    config: Any,
    samples: list[Any],
    initial_weights: Any,
    iterations: tuple[int, ...],
) -> dict[str, Replicated]:
    """Evaluate the same config across seeds and collect each metric by T.

    Returns one `Replicated` per `metric@T`, so a curve can be read with its
    seed spread rather than as a single line.
    """
    from flock.evaluation.probe import evaluate_by_iteration

    collected: dict[str, list[float]] = {}
    for run_dir in run_dirs:
        curves = evaluate_by_iteration(
            load_checkpoint(run_dir, config), samples, config, initial_weights, iterations
        )
        for name, curve in curves.items():
            for step, value in zip(iterations, curve.values, strict=True):
                collected.setdefault(f"{name}@{step}", []).append(float(value))
    return {key: Replicated(key, tuple(values)) for key, values in collected.items()}
