"""Backend port (spec §3.5, §7.3, ADR-0002).

Contracts are stated at *function* granularity, never per array operation. An
adapter implements these four calls in its own idiom, so `mx.compile` sees real
MLX operations with nothing of ours in the hot loop — the overhead that risk R5
warns about must not come from our own architecture.

The genuinely framework-neutral seam is the `.npz` on disk, not this protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from flock.domain.dynamics.state import CellConfig, CellState, StaticInputs

Params = Any
"""Backend-owned parameter tree. Opaque to the domain by design."""

OptState = Any
"""Backend-owned optimiser state."""


@runtime_checkable
class Backend(Protocol):
    """A compute adapter able to run and train the cell."""

    name: str

    def init_params(self, config: CellConfig, seed: int) -> Params:
        """Initialise cell parameters."""
        ...

    def rollout(
        self,
        params: Params,
        state: CellState,
        static: StaticInputs,
        steps: int,
    ) -> CellState:
        """Apply the cell `steps` times, sharing parameters across iterations.

        Unrolled rather than scanned: `steps` is small (2-4 under BPTT) and a
        static unroll keeps shapes constant for the compiler (spec §2.5).
        """
        ...

    def lbs(self, positions: Any, weights: Any, bones: Any, transforms: Any) -> Any:
        """Fused linear blend skinning; one einsum over precomputed transforms."""
        ...

    def train_step(
        self,
        params: Params,
        opt_state: OptState,
        batch: Any,
    ) -> tuple[Params, OptState, dict[str, float]]:
        """One optimiser step: k-step rollout, loss, gradients, update.

        The whole step is compiled as a unit. Implementations must evaluate
        lazily and synchronise exactly once, at the end (spec §3.5).
        """
        ...
