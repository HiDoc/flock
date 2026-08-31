"""The MLX backend adapter (ADR-0002).

Implements `flock.domain.dynamics.ports.Backend` by delegating to the modules in
this package. This class is the only thing that satisfies the port — the modules
themselves stay plain functions, because that is what `mx.compile` prefers to
be handed.
"""

from __future__ import annotations

from typing import Any

from flock.backends.mlx import cell, lbs, rollout, train_step
from flock.domain.dynamics.state import CellConfig, CellState, StaticInputs


class MlxBackend:
    """Apple Silicon backend, via MLX/Metal."""

    name = "mlx"

    def __init__(self, config: CellConfig) -> None:
        """Bind the backend to a cell configuration."""
        self._config = config
        self._compiled: Any = None
        self._optimizer: Any = None

    def init_params(self, config: CellConfig, seed: int) -> Any:
        """Initialise cell parameters."""
        return cell.init_params(config, seed)

    def rollout(
        self,
        params: Any,
        state: CellState,
        static: StaticInputs,
        steps: int,
    ) -> CellState:
        """Apply the cell `steps` times."""
        return rollout.rollout(params, state, static, steps, self._config)

    def lbs(self, positions: Any, weights: Any, bones: Any, transforms: Any) -> Any:
        """Fused linear blend skinning."""
        return lbs.lbs(positions, weights, bones, transforms)

    def train_step(
        self,
        params: Any,
        opt_state: Any,
        batch: Any,
    ) -> tuple[Any, Any, dict[str, float]]:
        """One compiled optimiser step.

        The compiled callable is built once and reused; rebuilding it per step
        would recompile the whole graph every time (spec §3.5).
        """
        if self._compiled is None:
            import mlx.optimizers as optim

            self._optimizer = optim.AdamW(learning_rate=3e-4)
            self._compiled = train_step.make_train_step(self._config, 4, self._optimizer)
        result: tuple[Any, Any, dict[str, float]] = self._compiled(params, opt_state, batch)
        return result
