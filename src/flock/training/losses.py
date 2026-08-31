"""Loss policy (spec §3.3).

Which losses run, at what relative weight, and when each switches on. The
arithmetic lives in `flock.backends.mlx.losses`; this module is the part that
states an experimental choice rather than a computation.

Two deliberate omissions.

Supervision lands at a *random* T drawn from the pool's age, and only at the
final step of the rollout. Supervising every step teaches the model to do all
its work in the first one and coast (spec R4) — the curve then goes flat after
T=1 and P1 is answered wrongly, in the flattering direction.

There is no Laplacian regulariser. Message passing is supposed to produce
smoothness on its own; if it does not, that is a finding about the architecture,
and smoothing it away would hide exactly the thing V0 exists to measure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    """Relative loss weights (spec §3.3)."""

    weight_l1: float = 1.0
    deformation: float = 1.0
    stability: float = 0.05
    support: float = 0.0
    """0 disables the optional support head; 0.1 enables it in V0.5."""

    deformation_warmup_steps: int = 1000
    """L_deform is held back until the field is roughly in place: posing a
    scrambled weight field produces a gradient about the scramble, not about
    the deformation."""

    def deformation_at(self, step: int) -> float:
        """The deformation weight in force at `step`."""
        return self.deformation if step >= self.deformation_warmup_steps else 0.0
