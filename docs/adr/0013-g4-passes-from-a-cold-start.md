# ADR-0013 — G4 passes: the cell solves from a cold start, on deformation only

**Status:** accepted · **Date:** 2026-09-02 · **Evidence:** G4, 3 seeds, 8 held-out meshes

## Context

Every result in this project until now started from a corrupted copy of the
ground truth — C1 damages ~11% of vertices, so the model began each evaluation
holding roughly 89% of the answer. That is the right protocol for P2 and P3, but
it left the obvious question untested: can this produce a skinning field at all?

Gate G4 answers it, and had never run, because it starts from **C3** and C3 was
deferred to V0.5 and never written.

C3 and C4 are now implemented. Both are *initialisations*, not corruptions: they
ignore the `weights` argument entirely and build a field from geometry alone.
That makes the anti-leakage rule R7 structural rather than procedural — C3
cannot carry ground truth into G4 because it never sees any — and it is asserted
by `test_c3_ignores_the_field_it_is_handed`, which feeds C3 the truth and a
zero field and requires identical output.

## Measured

From C3, whose weight L1 is **1.3563** — 8.5× further from the truth than C1's
0.1603.

| | deformation | weight L1 |
|---|---|---|
| B1 = C3 unrefined | 0.02448 | 1.2477 |
| B5, Reynolds tuned over 1,715 configs | 0.02165 | 1.1982 |
| **model (base 15k) @T=8** | **0.00647 ± 0.00058** | 0.6017 |

**GATE G4: PASS** — 3.78× better than B1, 3.35× better than B5.

`inverse_distance_weights` and C3 agree to 0.00e+00, so B1 and the starting
point are provably the same field and "beats B1" means exactly "improves on its
own input".

## Decisions

**1. G4 passes, and the qualifier travels with it.** Deformation improves 3.78×;
weight L1 only 2.07× (1.2477 → 0.6017), against the 0.0234 the same model
reaches from C1. **The cell substantially improves a bad guess; it does not
solve the problem.** Both halves are recorded, and neither is to be quoted
without the other.

The gap between the two ratios is itself the H3 prediction: the model finds a
*deformation-reasonable* field that is not the artist's convention, and the
control metric punishes it for that while the reference metric does not.

**2. B5 was given its best shot on the metric being compared.** The first tuning
optimised weight L1, which would have left the comparison open to the objection
that the fixed rules were never tuned for deformation. Re-running the full
1,715-config grid against deformation returns **the identical configuration**
(separation 1.0, alignment 0.0, cohesion 0.0, step 0.5) and the identical score.
The objection is closed rather than argued away.

Worth recording what that configuration is: the grid search **zeroes alignment
and cohesion**. "Three fixed rules" collapses to one — geodesic attenuation —
and improves the naive initialisation by 12% where the learned cell improves it
by 278%. §4.2's necessity test is answered decisively.

**3. From C3 the dynamics is stable, and this reframes G2.** *(Withdrawn by [ADR-0015](0015-the-dynamics-is-a-flow-not-an-attractor.md): no two trajectories converge over 2,048 steps, so C3 sits in a slow region rather than a basin. There is no attractor here to be in the wrong place.)* Deformation runs
0.00647 at T=8 → 0.00656 at T=32 (**1.4%**) and weight L1 0.6017 → 0.6016 —
comfortably inside G2's 5% criterion, where the same models starting from C1
fail at 1.097.

So the drift is not a property of the cell everywhere. **From a cold start it
converges to a genuine fixed point and stays there; the fixed point is simply
not the artist solution.** From near the ground truth it drifts away from it.
Two basins, and the attractor the dynamics actually owns sits at L1 ≈ 0.60.

That is a more precise statement of the P2 problem than "it drifts", and it
points the next experiment somewhere different: not at stabilising the dynamics,
which is stable, but at moving its attractor onto the solution.

## Consequences

Four gates now have verdicts: G0 PASS, G0.5 PASS, G1 PASS, **G4 PASS**. G2 and
G3 remain unresolved, and the reframing above is the most useful thing this gate
contributed to them.

The claim this licenses is narrow and should stay narrow: *a 33,665-parameter
recurrent cell beats a tuned hand-coded flocking baseline and the standard
geometric heuristic, from a cold start, on deformation error, on one 22-joint
skeleton.* It does not license "it can rig a character".
