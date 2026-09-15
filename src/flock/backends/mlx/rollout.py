"""Rollout — the cell applied T times with shared parameters (spec §2.5).

Unrolled with a Python loop rather than scanned: `steps` is small (k=4 under
truncated BPTT) and a static unroll keeps every shape constant, which is what
`mx.compile` needs to avoid silently recompiling.

The same parameters are applied at every step. That sharing *is* the hypothesis:
if unshared depth of the same size wins, P1 has failed (gate G1, ablation A2).
"""

from __future__ import annotations

from flock.backends.mlx.cell import Params, cell_step
from flock.domain.dynamics.state import CellConfig, CellState, StaticInputs


def rollout(
    params: Params,
    state: CellState,
    static: StaticInputs,
    steps: int,
    config: CellConfig,
) -> CellState:
    """Apply the cell `steps` times."""
    for step in range(steps):
        state = cell_step(params, state, static, config, step)
    return state


def rollout_trace(
    params: Params,
    state: CellState,
    static: StaticInputs,
    checkpoints: tuple[int, ...],
    config: CellConfig,
) -> dict[int, CellState]:
    """Roll out, capturing the state at each requested iteration count.

    Evaluation-only. Every metric in spec §4.1 is a function of T, so the
    evaluator needs the whole trajectory rather than its endpoint — a single
    number cannot tell "converged" from "still improving" from "drifting".
    """
    wanted = sorted(set(checkpoints))
    captured: dict[int, CellState] = {}
    if 0 in wanted:
        captured[0] = state
    for step in range(1, max(wanted) + 1 if wanted else 1):
        state = cell_step(params, state, static, config, step - 1)
        if step in wanted:
            captured[step] = state
    return captured
