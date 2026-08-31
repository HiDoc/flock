"""The dependency rule, enforced rather than documented (ADR-0001).

A layering convention that lives only in a README is a convention until the
first deadline. This walks the AST of every module and fails the build instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "flock"

# package prefix -> module prefixes it may NOT import.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "domain": ("mlx", "igl", "potpourri3d", "trimesh", "typer", "flock.data",
               "flock.backends", "flock.training", "flock.evaluation", "flock.cli"),
    "data": ("mlx", "flock.backends", "flock.training", "flock.evaluation", "flock.cli"),
    "backends": ("igl", "potpourri3d", "trimesh", "flock.training",
                 "flock.evaluation", "flock.cli"),
    "training": ("igl", "potpourri3d", "flock.cli"),
    "evaluation": ("igl", "potpourri3d", "flock.cli"),
    "experiment": ("mlx", "igl", "potpourri3d", "flock.cli"),
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _modules() -> list[tuple[str, Path]]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        layer = path.relative_to(SRC).parts[0]
        if layer in FORBIDDEN:
            out.append((layer, path))
    return out


@pytest.mark.parametrize(("layer", "path"), _modules(), ids=lambda x: getattr(x, "name", x))
def test_layer_does_not_import_forbidden_modules(layer: str, path: Path) -> None:
    for imported in _imports(path):
        for banned in FORBIDDEN[layer]:
            assert not (imported == banned or imported.startswith(banned + ".")), (
                f"{path.relative_to(SRC)} is in layer '{layer}' and must not import {imported!r}"
            )


def test_domain_layer_is_actually_covered() -> None:
    """Guard against the rule passing because it checked nothing."""
    assert sum(1 for layer, _ in _modules() if layer == "domain") >= 8
