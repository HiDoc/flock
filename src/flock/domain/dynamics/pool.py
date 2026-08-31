"""State pool (spec §3.1).

The pool is the mechanism behind P2 and P3. By feeding detached states back, the
model sees ages from 0 to ~100 iterations while gradients only ever flow through
k=4 steps. Without it there is no pressure to be stable *beyond* the supervised
horizon — which is exactly what ablation A3 tests.

Three fractions of every batch are rewritten, and each one buys a specific
property:

* **reset** with fresh corruption — keeps early-age states in circulation, so
  the dynamics never specialises to already-converged inputs;
* **clean** ground truth — the do-no-harm objective. Without it, collapsing to
  the identity is a winning strategy whenever inputs are usually near-correct,
  and it scores well on L1 while making repair impossible (R3);
* **recorrupt** an already-converged state — the only place P3 is actually
  trained. Repair cannot be learned from states that were never converged.

This module owns the bookkeeping and the state arrays. The arrays stay in NumPy:
the pool is written once per step outside the compiled region, and a scatter in
the hot path is exactly what §3.5 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flock.domain.skinning.corruption import CorruptionLevel

POOL_SIZE = 1024
BATCH_SIZE = 16
RESET_FRACTION = 1 / 8
"""Share of each batch reinitialised with fresh corruption."""
CLEAN_FRACTION = 1 / 16
"""Share reset to clean GT, so leaving good states alone is trained for (R3)."""
RECORRUPT_FRACTION = 1 / 8
"""Share of already-converged states corrupted again, to train repair (P3)."""
CONVERGED_AGE = 8
"""Iterations after which a state is considered settled enough to recorrupt."""


@dataclass(frozen=True)
class BatchPlan:
    """What to do with each sampled slot on this optimiser step.

    Attributes:
        slots: `[batch]` pool indices to read.
        reset: `[batch]` bool, reinitialise with fresh corruption.
        clean: `[batch]` bool, reset to uncorrupted ground truth.
        recorrupt: `[batch]` bool, apply a local corruption to a converged state.
    """

    slots: NDArray[np.int64]
    reset: NDArray[np.bool_]
    clean: NDArray[np.bool_]
    recorrupt: NDArray[np.bool_]


class StatePool:
    """Fixed-size pool of persistent cell states.

    Follows the six-step cycle of spec §3.1: sample, reset a fraction, recorrupt
    a fraction, roll out k steps with gradients, update, write states back
    *detached*.
    """

    def __init__(
        self,
        num_meshes: int,
        vertices: int,
        hidden_dim: int,
        candidates: int,
        size: int = POOL_SIZE,
        seed: int = 0,
    ) -> None:
        """Allocate the pool and assign each slot a mesh."""
        self.size = size
        self._rng = np.random.default_rng(seed)
        self.mesh_index = self._rng.integers(0, num_meshes, size).astype(np.int64)
        self.age = np.zeros(size, dtype=np.int64)
        self.hidden = np.zeros((size, vertices, hidden_dim), dtype=np.float32)
        self.logits = np.zeros((size, vertices, candidates), dtype=np.float32)
        self.level = np.array([CorruptionLevel.C0_GAUSSIAN_LOGITS] * size, dtype=object)
        # Every slot starts unwritten; the first batch resets whatever it draws.
        self.initialised = np.zeros(size, dtype=bool)

    def plan_batch(self, batch_size: int = BATCH_SIZE) -> BatchPlan:
        """Sample slots and decide which get reset, cleaned or recorrupted."""
        slots = self._rng.choice(self.size, size=batch_size, replace=False)
        draw = self._rng.random(batch_size)

        clean = draw < CLEAN_FRACTION
        reset = (draw >= CLEAN_FRACTION) & (draw < CLEAN_FRACTION + RESET_FRACTION)
        # Only a settled state can be recorrupted: repair is meaningless applied
        # to a state that had not converged in the first place.
        settled = self.age[slots] >= CONVERGED_AGE
        upper = CLEAN_FRACTION + RESET_FRACTION + RECORRUPT_FRACTION
        recorrupt = (draw >= CLEAN_FRACTION + RESET_FRACTION) & (draw < upper) & settled

        # An uninitialised slot must be written before it can be rolled out.
        fresh = ~self.initialised[slots]
        reset = (reset | fresh) & ~clean
        recorrupt = recorrupt & ~reset & ~clean
        return BatchPlan(slots, reset, clean, recorrupt)

    def read(self, slots: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Hidden state and logits for the sampled slots."""
        return self.hidden[slots], self.logits[slots]

    def write(
        self,
        slots: NDArray[np.int64],
        hidden: NDArray[np.float32],
        logits: NDArray[np.float32],
        steps: int,
    ) -> None:
        """Write detached states back and advance their age."""
        self.hidden[slots] = hidden
        self.logits[slots] = logits
        self.age[slots] += steps
        self.initialised[slots] = True

    def reset_slots(
        self,
        slots: NDArray[np.int64],
        logits: NDArray[np.float32],
        levels: list[CorruptionLevel],
    ) -> None:
        """Reinitialise slots with a fresh starting field, age back to zero.

        Levels are recorded per slot, not per group: which corruption a state
        started from is what the repair analysis reads back later.
        """
        self.hidden[slots] = 0.0
        self.logits[slots] = logits
        self.age[slots] = 0
        for slot, level in zip(slots, levels, strict=True):
            self.level[slot] = level
        self.initialised[slots] = True

    def level_counts(self) -> dict[str, float]:
        """How much of the pool currently sits at each corruption level."""
        live = self.level[self.initialised]
        total = max(live.size, 1)
        return {
            f"pool_{level.value}": float((live == level).sum()) / total
            for level in CorruptionLevel
            if (live == level).any()
        }

    def age_histogram(self) -> dict[str, float]:
        """Age spread, the diagnostic that says whether the pool is working.

        If ages stay near zero the pool is churning and long-horizon dynamics
        are never seen; if they all pile up at the maximum, nothing is being
        reset and the model only ever sees converged states.
        """
        live = self.age[self.initialised]
        if live.size == 0:
            return {"age_mean": 0.0, "age_max": 0.0, "age_p90": 0.0}
        return {
            "age_mean": float(live.mean()),
            "age_max": float(live.max()),
            "age_p90": float(np.percentile(live, 90)),
        }
