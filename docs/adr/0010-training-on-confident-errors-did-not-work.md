# ADR-0010 — Training on confident errors did not teach the model to see them

**Status:** superseded in part by [ADR-0012](0012-the-budget-was-the-confound.md) — at the full 15,000-step budget the intervention works (C1P repair 0.764). The diagnosis below stands; the verdict on the fix does not.
**Status:** accepted · **Date:** 2026-09-01 · **Evidence:** 3 seeds per arm, 5,000 steps

## Context

[ADR-0008](0008-repair-protocol-and-the-drift-control.md) established that the
cell detects *"this looks unsolved"* rather than *"this is wrong"*: repair
tracks the entropy of the damaged region and ignores neighbour disagreement
entirely. Every corruption the default curriculum asks it to fix arrives flat,
so peakedness is a near-perfect proxy for "needs work", and gradient descent
found it.

That diagnosis names its own fix. Spec §3.2 defines C1 as "weights randomised
**or permuted**" and only the randomised half was ever implemented. The permuted
half preserves a row's distribution exactly and moves it to the wrong bone — a
confidently wrong state, the thing the shortcut cannot see. It was added as
`C1P_PERMUTED_PATCH`, kept a separate level so results measured against C1 stay
comparable, and given half of C1's curriculum share. One change; everything else
matched to the existing three-seed arm.

Pre-registered baseline: the default arm repairs **−0.001 ± 0.002** of C1P.

## Measured

Pool composition confirms the curriculum did what was asked (C1P at 16% of
resident states, C1 halved from 31% to 16%, clean unchanged at ~45%).

| | default (3 seeds) | confident-error (3 seeds) |
|---|---|---|
| repaired @C1P | −0.001 ± 0.002 | **0.004 ± 0.006** |
| moved inside @C1P | 0.090 ± 0.042 | 0.118 ± 0.050 |
| repaired @C1 | 0.505 ± 0.169 | 0.428 ± 0.064 |
| repaired @C2 | −0.001 ± 0.001 | −0.001 ± 0.001 |
| weight L1 @T=8 | **0.0321 ± 0.0019** | 0.0403 ± 0.0026 |
| deformation @T=8 | **0.00038 ± 0.00003** | 0.00050 ± 0.00005 |
| drift T=32/T=8 | 1.223 ± 0.058 | **1.083 ± 0.056** |

Extending the horizon does not rescue it: C1P repair reaches 0.007 at T=32
against a 0.80 threshold, and the state moves 0.133 inside the region against
C1's ~1.0.

## Decisions

**1. The fix failed, and it is recorded as a failure.** Training directly on
confident errors, at 16% of the pool for 2,000 steps, moved repair from −0.001
to 0.004 where 0.80 is needed. The model did not learn to see them.

**2. The cost is real and disqualifying on its own.** Weight L1 is 26% worse and
deformation — the reference metric — is 32% worse. Even had repair improved
slightly, this arm is not the better model.

**3. The one thing that improved was not the target.** Drift fell from 1.223 to
1.083 against a 1.05 threshold: still a G2 failure, but the largest movement any
intervention has produced on it. Whether that is the corruption's doing or a
side effect of the accuracy regression (a worse model has more room to improve
before it degrades) is **not established** — the ratio is scale-free but not
immune to that confound, and no experiment here separates them.

## Consequences

The entropy diagnosis of ADR-0008 stands — it was a measurement, and this result
does not touch it. What is refuted is the inference drawn from it, that a
curriculum of confident errors would teach the model to judge correctness.

Three readings remain open, in the order they should be tested:

* **Underpowered.** C1P appeared only from step 3,000, so it got 2,000 steps at
  16%. This was flagged before the run. A longer run, or moving C1P into the
  first stage, is the cheapest discriminator and must come before any structural
  conclusion.
* **The signal is outvoted.** Clean states are ~45% of the pool and are peaked
  by construction, so nearly half the training signal says *leave peaked states
  alone* against 16% saying *fix these peaked states*. The conflict is 3:1
  against the new signal, and entropy resolves it in the majority's favour.
* **The information is not reachable.** Judging a peaked row wrong requires
  comparing it against geometry rather than against its own shape. The pair
  features carry that geometry, but nothing forces the cell to use it while an
  easier signal remains available.

The second reading is testable without new machinery — vary the clean fraction —
and would explain both this result and the original shortcut with one mechanism.
It is the next thing to run.

No ablation here changes P3's verdict: G3 still fails, and repair of confidently
wrong states remains unachieved.
