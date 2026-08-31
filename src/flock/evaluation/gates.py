"""Go / no-go gates (spec §6).

The gates are the point of V0. Each is a threshold fixed *before* the experiment
runs, so that the answer cannot drift to meet the result — and G1 is
eliminatory: fail it, and the project is "another GNN" and must pivot or stop.

Encoded as data rather than prose so the CLI can evaluate them, print a verdict,
and write it into the run directory alongside the metrics that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GateStatus(StrEnum):
    """Outcome of a gate evaluation."""

    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class Gate:
    """One quantified decision point."""

    key: str
    question: str
    criterion: str
    on_failure: str


GATES: tuple[Gate, ...] = (
    Gate(
        key="G0",
        question="Do the B candidates contain the ground-truth weight mass?",
        criterion="coverage mean >= 0.98 and p5 >= 0.90",
        on_failure="Fix candidate selection. Train nothing: this ceiling is "
        "unreachable by the model (R1) and would contaminate every conclusion.",
    ),
    Gate(
        key="G0.5",
        question="Does the loop learn at all?",
        criterion="single-mesh overfit reaches L1 ~ 0 in < 10 min",
        on_failure="Loop or gradient bug. Stop and fix before anything else.",
    ),
    Gate(
        key="G1",
        question="Is the recurrence useful? (P1)",
        criterion="L1(T=8) <= 0.8 * L1(T=1) and within 5% of B3 with ~8x fewer parameters",
        on_failure="Two days maximum of fixes — smaller alpha, random T, pool, "
        "fire rate. Then reject P1 and pivot to feed-forward + refinement, or stop.",
    ),
    Gate(
        key="G2",
        question="Is there an attractor? (P2)",
        criterion="degradation < 5% between T=8 and T=32, median stability decreasing",
        on_failure="Work on L_stab and A7. If still unstable, self-organisation is "
        "weakened but not dead — document it.",
    ),
    Gate(
        key="G3",
        question="Does it repair? (P3)",
        criterion=">= 80% of induced error in the corrupted region resolved within "
        "8 steps, collateral damage < 10% relative",
        on_failure="Check the do-no-harm fraction and pool recorruption before concluding.",
    ),
    Gate(
        key="G4",
        question="Is it useful at all?",
        criterion="deformation error below B1 and below B5 (hand-coded Reynolds), from C3",
        on_failure="Beaten by a heuristic or by three fixed rules means learning the "
        "cell is not justified. Insufficient grounds for V1.",
    ),
)


@dataclass(frozen=True)
class GateResult:
    """A gate evaluated against measured metrics."""

    gate: Gate
    status: GateStatus
    measured: dict[str, float]
    note: str = ""


def evaluate_gate(key: str, metrics: dict[str, float]) -> GateResult:
    """Evaluate one gate against a metrics dictionary.

    Raises:
        NotImplementedError: G0 lands at M1; the rest at M4.
    """
    raise NotImplementedError("M1 (G0) / M4 (G1-G4): gate evaluation")
