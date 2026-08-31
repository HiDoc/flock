"""Backend selection (ADR-0002).

One lookup, imported lazily so that a machine without MLX can still run the
domain tests and the offline preprocessing. Spec §7.3 wants the code portable if
a compute wall ever forces GPU rental; a second backend lands here as one more
branch and one more sibling package.
"""

from __future__ import annotations

from flock.domain.dynamics.ports import Backend
from flock.domain.dynamics.state import CellConfig

AVAILABLE = ("mlx",)


def get_backend(name: str, config: CellConfig) -> Backend:
    """Instantiate a backend by name.

    Args:
        name: One of `AVAILABLE`.
        config: Cell configuration to bind.

    Returns:
        A `Backend` implementation.

    Raises:
        ValueError: If `name` is not a known backend.
    """
    if name == "mlx":
        from flock.backends.mlx.backend import MlxBackend

        return MlxBackend(config)
    raise ValueError(f"unknown backend {name!r}; available: {', '.join(AVAILABLE)}")
