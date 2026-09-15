# ADR-0012 — The training budget was the confound

**Status:** accepted · **Date:** 2026-09-02 · **Evidence:** 9 runs, 3 arms × 3 seeds, 15,000 steps

## Context

Every result in this repo up to now came from 5,000-step runs — a third of spec
§3.4's 10–15k budget. Two consequences went unnoticed until late:

* **C2 never entered training.** `default_curriculum` introduces it at step
  8,000, so no model had ever seen it when G3 declared C2 unrepairable.
* **C1P got 2,000 steps** in the confident-error arm, which
  [ADR-0010](0010-training-on-confident-errors-did-not-work.md) flagged as a
  power risk before concluding the intervention had failed.

Three arms were run at the full budget, each one change from its neighbour:
`base` (default curriculum), `ce` (+C1P at half of C1's share), `ce-lowclean`
(+clean share 0.42 → 0.25, testing ADR-0010's outvoting hypothesis).

## Measured

| | base 5k | base 15k | ce 15k | ce-lowclean 15k |
|---|---|---|---|---|
| weight L1 @T=8 | 0.0321 | **0.0234** | 0.0265 | 0.0277 |
| deformation @T=8 | 0.00038 | **0.00031** | 0.00034 | 0.00037 |
| drift L1 T=32/T=8 | 1.223 | **1.097** | 1.112 | 1.193 |
| drift deformation | 1.153 | **1.048** | 1.245 | 1.407 |
| repaired @C1 | 0.505 | 0.785 | **0.807** | 0.818 |
| repaired @C2 | −0.001 | −0.001 | **0.660** | 0.436 ± 0.390 |
| repaired @C1P | −0.001 | −0.002 | **0.764** | 0.524 ± 0.456 |
| moved inside @C1P | 0.090 | 0.065 | **1.524** | 1.046 |

## Decisions

**1. ADR-0010's conclusion is withdrawn. The intervention works.** Repair of
confident errors goes from −0.002 to **0.764 ± 0.041**, and engagement inside
the damaged region from 0.065 to **1.524** — a 23× increase. The 5,000-step null
was underpowering, exactly the risk that was flagged and then reasoned past.
ADR-0010's *diagnosis* (the entropy shortcut) stands; its verdict on the fix
does not.

**2. C1P teaches what C2 cannot.** This is the finding worth keeping. `base 15k`
trained on C2 for 7,000 steps and still repairs it at **−0.001**. `ce 15k`, which
saw the same C2 plus C1P, repairs C2 at **0.660**. Training directly on a
corruption does not teach the model to see it; training on the *permuted* one
does, and the capability then transfers. C1P preserves a row's distribution
exactly, so the only thing wrong is the assignment — an unambiguous "confidently
wrong" signal, where C2 confounds the assignment error with a change in shape.

**3. Much of the G2 drift was under-convergence.** Weight-L1 drift falls 1.223 →
1.097 and deformation drift 1.153 → **1.048**, which is under the 1.05
threshold. But the seeds straddle it — [1.032, 1.012, 1.099] — so by this
project's own rule a mean on the right side of a threshold its seeds disagree
about **has not decided anything**. G2 is recorded as *inconclusive on
deformation, still failing on weight L1*, not as a pass.

**4. The outvoting hypothesis is refuted.** Lowering the clean share made every
repair metric worse and wildly unstable (C2 0.660 → 0.436 ± 0.390, C1P 0.764 →
0.524 ± 0.456). The clean fraction was not suppressing the signal; removing it
destabilised training. ADR-0010's leading reading was wrong.

## Consequences

No gate flips cleanly, and none is claimed. G3's C1 repair reaches 0.807 ± 0.019
for `ce 15k` with attributable collateral of +0.048 — both halves nominally met
— but the seeds straddle 0.80, so it is not decided either.

What did change is the standing of every earlier verdict. **G1's PASS is the
only conclusion here still resting on matched arms**, because both it and B3
were trained at 5k; the 15k shared model is 27% better on weight L1 and has no
control. B3 at 15k is required before G1 can be restated at this budget.

The methodological lesson is cheap to state and was expensive to learn: an
under-budget run is not a small version of the real experiment. It silently
removed a curriculum stage and turned a working intervention into a null.
Budget belongs in the comparison table beside the metric, and no arm should be
read against another trained for a different number of steps.
