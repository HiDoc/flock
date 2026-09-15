# ADR-0015 — It is a monotone flow, not an attractor in the wrong place

**Status:** accepted · **Date:** 2026-09-03 · **Evidence:** 3 seeds × 3 meshes, horizons to 2,048 steps

## Context

G2 has been failing since it was first measured, and the diagnosis kept moving:
accumulated noise ([ADR-0009](0009-the-drift-is-constant-velocity.md), then
withdrawn), a growing instability (withdrawn from the same ADR), delta
accumulation (ruled out by A4 in
[ADR-0011](0011-a4-damped-deltas-confirmed.md)), and most recently a *stable
wrong attractor* ([ADR-0013](0013-g4-passes-from-a-cold-start.md)).

The NCA literature suggested a fifth reading. Kvalsund & Stovold
([arXiv 2604.12720](https://arxiv.org/abs/2604.12720)) find that NCAs exhibit
oscillatory and quasi-periodic behaviour rather than fixed points, "challenging
the belief that NCAs learn fixed point attractors" — which would mean P2's
criterion, not our model, was at fault. An oscillation and a drift fail G2
identically and need opposite fixes, so the character of the motion had to be
measured before anything else was tried.

## Method, and one thing that did not work

A largest-Lyapunov estimate was attempted and **abandoned**. State crosses the
backend port as NumPy and is therefore quantised to float32 every step, and a
two-trajectory estimate never entered a linear regime: the exponent scaled with
the perturbation instead of converging (0.34, 0.16, 0.07 for eps 3e-4, 1e-3,
3e-3). An earlier run with eps=1e-5 spread the perturbation over ~41,000
dimensions, putting each component near 5e-8 — below float32 epsilon — and
returned a confident λ ≈ +1.0 that was pure numerical noise.

The measures kept are large-amplitude and quantisation-robust, and live in
`evaluation/dynamics.py`.

## Measured

3 seeds × 3 meshes, 1,024 steps past T=32:

| | mean | range |
|---|---|---|
| cos(consecutive step directions) | **0.999 ± 0.001** | [0.998, 1.000] |
| PC1 share of trajectory variance | 0.962 ± 0.029 | [0.920, 0.996] |
| components for 95% variance | 1.44 ± 0.53 | [1, 2] |
| error growth from a **ground-truth** start | **52.9× ± 28.5** | [33.8, 120.8] |
| error growth from a C1 start | 4.7× ± 2.1 | [1.9, 8.5] |

Pairwise, over 2,048 steps on one mesh, no two trajectories converge:

| | separation start → end |
|---|---|
| GT vs C1 | 0.0293 → 0.0388 |
| GT vs C3 | 0.8699 → 0.8318 |
| C1 vs C3 | 0.8566 → 0.8180 |
| C3 vs C4 | 0.6401 → 0.5800 |

## Decisions

**1. It is a smooth, near-one-dimensional monotone flow.** Successive step
directions agree to 0.999 and one principal component carries 96% of the
trajectory's variance. There is no oscillation, no limit cycle, no torus.

**2. The NCA reading does not apply to FlockSkin, and P2's criterion is
vindicated.** Kvalsund & Stovold's oscillatory finding was the one available
route to concluding that G2 asks for something this architecture family cannot
give. It is not what our system does. **G2's failure is real, not a
mis-specified gate.** That closes off the comfortable interpretation.

**3. ADR-0013's "stable wrong attractor" is withdrawn.** No pair of
trajectories converges — GT and C3 sit 0.87 apart at step 0 and 0.83 apart at
step 2,048. There are no basins. C3 appeared stable only because it starts
where the flow is slowest (0.8707 → 0.8882, 2% over 2,048 steps); slowness is
not an attractor.

**4. G2 fails far worse than the T=32 ratio implied.** Handed the *ground
truth*, the cell moves away from it by **53× ± 28** over 1,024 steps, reaching
weight L1 0.186 by step 2,079 on one mesh — worse than the C1 corruption
(0.160) it exists to repair. Given the right answer, it eventually destroys it.
The published 1.097 at T=32 was the first 3% of that curve.

**5. This weakens FlowDPO and strengthens DEQ.** FlowDPO
([arXiv 2606.29150](https://arxiv.org/abs/2606.29150)) deepens the basin around
the correct solution by mining confident wrong *attractors*; measurement 3 says
this system has no basins to deepen, so the premise does not hold here. That
retracts the recommendation made before this measurement was run.

Conversely, if training at k=4 with one overflow step yields a flow rather than
a fixed point, then a fixed point has to be **imposed** rather than encouraged —
which is what deep equilibrium models
([arXiv 1909.01377](https://arxiv.org/abs/1909.01377)) do by construction. The
cost stays what it was: DEQ guarantees only the equilibrium is a solution, and
A4 showed that *every intermediate state being a solution* is this design's
distinguishing strength.

## Consequences

P1 and P3 are unaffected. P2 now has a specific, replicated mechanism rather
than a moving diagnosis, and the next experiment is no longer a choice among
five readings.

The measurement cost was hours and no training, against the day-plus that either
candidate treatment would have taken — and both treatments were mis-aimed before
it. That ordering is the transferable part.
