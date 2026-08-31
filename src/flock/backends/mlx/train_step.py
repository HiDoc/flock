"""The compiled training step (spec §3.5).

The whole step — k-step rollout, loss, gradients, optimiser update — is compiled
as one unit. The rules that keep it fast, in the order they matter:

* fixed shapes everywhere, or the compiler silently recompiles;
* gather only, never scatter: `[V, K]` and `[V, B]` are constant index tables,
  so aggregation is a `take` plus a reduction;
* exactly one `mx.eval` per optimiser step, and no `.item()` or print inside the
  loop — each one is a GPU synchronisation;
* the dataset lives resident in unified memory, so there is no loader and no
  host-device copy at all.

Risk R5 says throughput here dies of Python overhead, not arithmetic. If the
step rate disappoints, look for a stray varying shape before optimising
anything else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim

from flock.backends.mlx import losses
from flock.backends.mlx.cell import Params, weights_of
from flock.backends.mlx.rollout import rollout
from flock.domain.dynamics.state import CellConfig, CellState, StaticInputs


def masked_weight_l1(predicted: mx.array, target: mx.array, vertex_mask: mx.array) -> mx.array:
    """Mean per-vertex L1 over real vertices only.

    Padded rows are excluded rather than averaged in; letting them count would
    scale the loss by the padding ratio and make meshes of different sizes
    incomparable.
    """
    per_vertex = mx.sum(mx.abs(predicted - target), axis=-1)
    return mx.sum(per_vertex * vertex_mask) / mx.maximum(mx.sum(vertex_mask), 1.0)


def make_full_loss(config: CellConfig, steps: int, overflow: int = 1) -> Callable[..., Any]:
    """Loss over a `steps` rollout, with `overflow` unsupervised steps past it.

    The overflow steps carry only `L_stab`. They are where the fixed point has
    to hold and where nothing else is pushing it to: the supervised step teaches
    the answer, these teach staying there.

    Supervision lands at the *final* rollout step only. Grading every step
    teaches the model to do all its work in the first one and coast (R4), which
    flattens the per-iteration curve and would answer P1 in the flattering
    direction.
    """

    def loss_fn(
        params: Params,
        batch: dict[str, mx.array],
        static_tree: dict[str, mx.array],
    ) -> tuple[mx.array, tuple[Any, ...]]:
        static = StaticInputs(**static_tree)
        state = CellState(hidden=batch["hidden"], logits=batch["logits"])

        final = rollout(params, state, static, steps, config)
        predicted = weights_of(final, config, static)
        mask = static.vertex_mask

        l_weight = losses.weight_loss(predicted, batch["target"], mask)
        l_deform = losses.deformation_loss(
            batch["positions"], predicted, batch["target"], batch["bones"], batch["poses"]
        )

        drifted = rollout(params, final, static, overflow, config)
        l_stability = losses.stability_loss(
            predicted, weights_of(drifted, config, static), mask
        )

        total = (
            batch["w_weight"] * l_weight
            + batch["w_deform"] * l_deform
            + batch["w_stability"] * l_stability
        )
        return total, (final.hidden, final.logits, l_weight, l_deform, l_stability)

    return loss_fn


def make_loss(config: CellConfig, steps: int) -> Callable[..., Any]:
    """Build the loss over a `steps`-long rollout."""

    def loss_fn(
        params: Params,
        batch: dict[str, mx.array],
        static_tree: dict[str, mx.array],
    ) -> mx.array:
        static = StaticInputs(**static_tree)
        state = CellState(
            hidden=mx.zeros(
                (batch["initial_logits"].shape[0], batch["initial_logits"].shape[1],
                 config.hidden_dim)
            ),
            logits=batch["initial_logits"],
        )
        final = rollout(params, state, static, steps, config)
        predicted = weights_of(final, config, static)
        return masked_weight_l1(predicted, batch["target"], static.vertex_mask)

    return loss_fn


def make_pool_train_step(
    config: CellConfig,
    steps: int,
    optimizer: optim.Optimizer,
    overflow: int = 1,
    compile_step: bool = True,
) -> Callable[..., Any]:
    """The M3 optimiser step: pooled states in, detached states out.

    Returns `(params, batch, static) -> (params, total, hidden, logits, parts)`.
    The final state comes back so the pool can write it, which is what lets
    gradients span 4 steps while the dynamics spans a hundred (§3.1).
    """
    loss_fn = make_full_loss(config, steps, overflow)
    # MLX differentiates the first element and passes the rest through; there is
    # no has_aux flag, the tuple return is the mechanism.
    value_and_grad = mx.value_and_grad(loss_fn)
    state = [optimizer.state]

    def step(
        params: Params,
        batch: dict[str, mx.array],
        static_tree: dict[str, mx.array],
    ) -> tuple[Params, mx.array, mx.array, mx.array, tuple[Any, ...]]:
        (total, aux), grads = value_and_grad(params, batch, static_tree)
        updated: Params = optimizer.apply_gradients(grads, params)
        hidden, logits, l_weight, l_deform, l_stability = aux
        return updated, total, hidden, logits, (l_weight, l_deform, l_stability)

    if not compile_step:
        return step
    return mx.compile(step, inputs=state, outputs=state)


def make_train_step(
    config: CellConfig,
    steps: int,
    optimizer: optim.Optimizer,
    compile_step: bool = True,
) -> Callable[..., Any]:
    """Build the M2 sanity-gate step, compiled as one unit.

    Returns a callable `(params, batch, static) -> (params, loss)`. The
    optimiser's own state is carried through `mx.compile`'s `inputs`/`outputs`,
    which is what lets Adam's moments live inside the compiled graph rather than
    forcing a synchronisation every step.
    """
    loss_fn = make_loss(config, steps)
    value_and_grad = mx.value_and_grad(loss_fn)
    state = [optimizer.state]

    def step(
        params: Params,
        batch: dict[str, mx.array],
        static_tree: dict[str, mx.array],
    ) -> tuple[Params, mx.array]:
        loss, grads = value_and_grad(params, batch, static_tree)
        updated: Params = optimizer.apply_gradients(grads, params)
        return updated, loss

    if not compile_step:
        return step
    return mx.compile(step, inputs=state, outputs=state)
