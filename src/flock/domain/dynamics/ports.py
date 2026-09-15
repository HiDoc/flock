"""Backend port (spec §3.5, §7.3, ADR-0002).

Contracts are stated at *function* granularity, never per array operation. An
adapter implements these four calls in its own idiom, so `mx.compile` sees real
MLX operations with nothing of ours in the hot loop — the overhead that risk R5
warns about must not come from our own architecture.

The boundary sits where it does for a specific reason. An earlier version of
this port exposed `rollout`, `lbs` and a `train_step` taking device arrays,
which left the training loop constructing `mx.array`s and calling `mx.eval`
itself — so the loop was MLX code wearing a protocol, and the port was
decorative. Everything crossing this interface is now **NumPy or opaque**: the
device exists only on the far side.

The genuinely framework-neutral seam is still the `.npz` on disk. This protocol
is what keeps a second backend from having to rewrite the training loop as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from flock.domain.dynamics.state import CellConfig

Params = Any
"""Backend-owned parameter tree. Opaque by design — the domain never reads it."""


@dataclass(frozen=True)
class RolloutFrame:
    """One captured iteration of the dynamics, host-side.

    The hidden state crosses the port as NumPy for one reason: P3 requires
    damaging a *converged* state and resuming from it, and a repair that left
    the hidden state intact inside the damaged region would be the cell reading
    its own memory of the answer rather than repairing anything.

    Attributes:
        weights: `[N, V, B]` weights after the softmax over candidates.
        hidden: `[N, V, H]` hidden state.
        logits: `[N, V, B]` candidate logits.
    """

    weights: NDArray[np.float32]
    hidden: NDArray[np.float32]
    logits: NDArray[np.float32]


@runtime_checkable
class Bank(Protocol):
    """Training meshes held in device memory, gathered by index.

    Resident rather than streamed: unified memory means the whole tier fits with
    no loader and no host-device copy (spec §3.5). A backend with discrete
    memory would implement this very differently, which is why it is a port call
    and not a shared helper.
    """

    def __len__(self) -> int:
        """Number of meshes held."""
        ...

    def gather(
        self,
        indices: NDArray[np.int64],
        pose_indices: NDArray[np.int64] | None = None,
    ) -> tuple[Any, Any]:
        """Static inputs and targets for the given meshes, device-side."""
        ...


@runtime_checkable
class PoolStep(Protocol):
    """One compiled optimiser step over pooled states."""

    def __call__(
        self,
        params: Params,
        hidden: NDArray[np.float32],
        logits: NDArray[np.float32],
        static: Any,
        extras: Any,
        loss_weights: dict[str, float],
    ) -> tuple[Params, dict[str, float], NDArray[np.float32], NDArray[np.float32]]:
        """Roll out k steps, take one optimiser step, return detached states.

        States come back as NumPy because the pool writes them on the host: the
        write is a scatter, and §3.5 keeps scatters out of the compiled region.
        Metrics come back as floats, which forces the single synchronisation per
        step that §3.5 allows.
        """
        ...


@runtime_checkable
class Backend(Protocol):
    """A compute adapter able to run and train the cell."""

    name: str

    def init_params(self, config: CellConfig, seed: int, depth: int = 1) -> Params:
        """Initialise cell parameters.

        `depth` is the rollout length an unshared model is trained at; it
        is ignored when weights are shared.
        """
        ...

    def make_bank(self, samples: list[Any]) -> Bank:
        """Load meshes into device memory."""
        ...

    def load_params(self, checkpoint: Any) -> Params:
        """Read a parameter tree from a checkpoint written by this backend."""
        ...

    def count_params(self, params: Params) -> int:
        """Total number of parameters in a tree.

        The tree is opaque above this line, so its size has to be asked for
        rather than computed.
        """
        ...

    def make_pool_step(
        self,
        config: CellConfig,
        bptt_steps: int,
        learning_rate: float,
        weight_decay: float,
        overflow: int = 1,
    ) -> PoolStep:
        """Build the compiled training step.

        Built once and reused; rebuilding per step would recompile the whole
        graph every time (spec §3.5).
        """
        ...

    def trace_from(
        self,
        params: Params,
        hidden: NDArray[np.float32] | None,
        logits: NDArray[np.float32],
        static: Any,
        checkpoints: tuple[int, ...],
        config: CellConfig,
    ) -> dict[int, RolloutFrame]:
        """Roll out from a given state, capturing each requested iteration.

        Evaluation-only. Every metric in spec §4.1 is a function of T, so the
        evaluator needs the whole trajectory rather than its endpoint.

        `hidden` of `None` starts from zeros, the fresh-state case. Passing a
        state back in is what lets an evaluation stop the dynamics, damage it,
        and resume — the protocol gate G3 needs.
        """
        ...
