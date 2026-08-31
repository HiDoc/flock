"""Run repository (spec §8).

Discipline, made mechanical: one change per run, `v0_<date>_<change>` naming,
config and metrics serialised next to every checkpoint, three seeds before any
conclusion.

None of that is novel — it is just the part of research hygiene that erodes
first under time pressure, which is why it lives in code rather than in a habit.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RUN_NAME_PATTERN = re.compile(r"^v[01](?:\.5)?_\d{8}_[a-z0-9]+(?:-[a-z0-9]+)*$")
SEEDS_PER_CONCLUSION = 3


@dataclass(frozen=True)
class RunRecord:
    """One experiment run on disk."""

    name: str
    path: Path
    config: dict[str, object]
    metrics: dict[str, object]


def make_run_name(version: str, change: str, today: dt.date | None = None) -> str:
    """Build a run name of the form `v0_20260831_delta-vs-absolute`.

    Args:
        version: `"v0"`, `"v0.5"` or `"v1"`.
        change: The single thing this run changes, in kebab-case.
        today: Date override, for tests.

    Returns:
        The run name.

    Raises:
        ValueError: If the resulting name is malformed.
    """
    stamp = (today or dt.date.today()).strftime("%Y%m%d")
    name = f"{version}_{stamp}_{change}"
    if not RUN_NAME_PATTERN.match(name):
        raise ValueError(f"malformed run name {name!r}; expected v0_YYYYMMDD_kebab-case")
    return name


class RunRepository:
    """Creates and reads run directories under a root."""

    def __init__(self, root: Path) -> None:
        """Bind the repository to a runs directory."""
        self.root = root

    def path_for(self, name: str) -> Path:
        """The directory holding one run."""
        return self.root / name

    def create(self, name: str, config: dict[str, object]) -> Path:
        """Create a run directory and write its config.

        The config is written *before* the first step, so an interrupted run
        still says what it was trying to do.
        """
        if not RUN_NAME_PATTERN.match(name):
            raise ValueError(f"malformed run name {name!r}; expected v0_YYYYMMDD_kebab-case")
        path = self.path_for(name)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(config, indent=2, default=str))
        (path / "metrics.jsonl").touch()
        return path

    def record_metrics(self, name: str, metrics: dict[str, object]) -> None:
        """Append one metrics record.

        JSON Lines, appended and flushed per record: a run killed mid-flight
        keeps everything it had measured up to that point.
        """
        with (self.path_for(name) / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics, default=str) + "\n")

    def save_checkpoint(self, name: str, params: dict[str, dict[str, Any]]) -> Path:
        """Write the parameter tree beside the config that produced it."""
        path = self.path_for(name) / "checkpoint.npz"
        flat = {
            f"{group}.{key}": np.asarray(value)
            for group, entries in params.items()
            for key, value in entries.items()
        }
        # numpy's stub folds **kwds into the allow_pickle overload; the call is fine.
        np.savez(str(path), **flat)  # type: ignore[arg-type]
        return path

    def load(self, name: str) -> RunRecord:
        """Read a run's config and every metrics record it wrote."""
        path = self.path_for(name)
        config = json.loads((path / "config.json").read_text())
        records = [
            json.loads(line)
            for line in (path / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        return RunRecord(name=name, path=path, config=config, metrics={"records": records})

    def list_runs(self) -> list[str]:
        """Run names, newest first."""
        if not self.root.exists():
            return []
        runs = [p for p in self.root.iterdir() if (p / "config.json").exists()]
        return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]
