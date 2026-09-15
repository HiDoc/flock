"""The MLX backend adapter (ADR-0002).

Implements `flock.domain.dynamics.ports.Backend` by delegating to the modules in
this package. This class is the only thing that satisfies the port — the modules
themselves stay plain functions, because that is what `mx.compile` prefers to
be handed.

Every value crossing back out is NumPy or a float. Nothing above this line ever
holds an `mx.array`, which is what makes the port load-bearing rather than
decorative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from numpy.typing import NDArray

from flock.backends.mlx import cell, rollout, train_step
from flock.backends.mlx.bank import MlxBank
from flock.domain.dynamics.ports import RolloutFrame
from flock.domain.dynamics.state import CellConfig, CellState, StaticInputs
from flock.domain.sample import PreprocessedMesh


class MlxBackend:
    """Apple Silicon backend, via MLX/Metal."""

    name = "mlx"

    def __init__(self, config: CellConfig) -> None:
        """Bind the backend to a cell configuration."""
        self._config = config

    def init_params(self, config: CellConfig, seed: int, depth: int = 1) -> Any:
        """Initialise cell parameters."""
        return cell.init_params(config, seed, depth)

    def make_bank(self, samples: list[PreprocessedMesh]) -> MlxBank:
        """Load meshes into unified memory."""
        return MlxBank(samples)

    def load_params(self, checkpoint: Path) -> Any:
        """Read a parameter tree from a `.npz` checkpoint."""
        params: dict[str, dict[str, mx.array]] = {}
        with np.load(checkpoint) as data:
            for flat_key in data.files:
                group, key = flat_key.split(".", 1)
                params.setdefault(group, {})[key] = mx.array(data[flat_key])
        return params

    def count_params(self, params: Any) -> int:
        """Total number of parameters in a tree."""
        return int(sum(v.size for group in params.values() for v in group.values()))

    def make_pool_step(
        self,
        config: CellConfig,
        bptt_steps: int,
        learning_rate: float,
        weight_decay: float,
        overflow: int = 1,
    ) -> Any:
        """Build the compiled training step, wrapped to speak NumPy."""
        optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        compiled = train_step.make_pool_train_step(config, bptt_steps, optimizer, overflow)

        def step(
            params: Any,
            hidden: NDArray[np.float32],
            logits: NDArray[np.float32],
            static: dict[str, mx.array],
            extras: dict[str, mx.array],
            loss_weights: dict[str, float],
        ) -> tuple[Any, dict[str, float], NDArray[np.float32], NDArray[np.float32]]:
            payload = {
                "hidden": mx.array(hidden),
                "logits": mx.array(logits),
                "w_weight": mx.array(loss_weights["weight"]),
                "w_deform": mx.array(loss_weights["deform"]),
                "w_stability": mx.array(loss_weights["stability"]),
                **extras,
            }
            params, total, new_hidden, new_logits, parts = compiled(params, payload, static)
            # Converting the states to NumPy is the single synchronisation per
            # step that §3.5 allows; the metrics ride along on it for free.
            host_hidden = np.array(new_hidden)
            host_logits = np.array(new_logits)
            metrics = {
                "loss": float(total),
                "l_weight": float(parts[0]),
                "l_deform": float(parts[1]),
                "l_stability": float(parts[2]),
            }
            return params, metrics, host_hidden, host_logits

        return step

    def trace_from(
        self,
        params: Any,
        hidden: NDArray[np.float32] | None,
        logits: NDArray[np.float32],
        static: dict[str, mx.array],
        checkpoints: tuple[int, ...],
        config: CellConfig,
    ) -> dict[int, RolloutFrame]:
        """Roll out from a given state, capturing each requested iteration."""
        inputs = StaticInputs(**static)
        state = CellState(
            hidden=(
                mx.zeros((logits.shape[0], logits.shape[1], config.hidden_dim))
                if hidden is None
                else mx.array(hidden)
            ),
            logits=mx.array(logits),
        )
        captured = rollout.rollout_trace(params, state, inputs, checkpoints, config)
        return {
            step: RolloutFrame(
                weights=np.array(cell.weights_of(captured[step], config, inputs)),
                hidden=np.array(captured[step].hidden),
                logits=np.array(captured[step].logits),
            )
            for step in checkpoints
        }
