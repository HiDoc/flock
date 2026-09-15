# ADR-0014 — A6 settles the candidate metric, and does not test H1

**Status:** accepted · **Date:** 2026-09-03 · **Evidence:** A6 and B3, 3 seeds each, 15,000 steps

## Context

Two things were outstanding at the spec budget: the last vital ablation (A6,
diffusion versus Euclidean candidates) and G1's control, which was stale because
the shared arm had moved to 15k while B3 sat at 5k.

Both were run overnight on identical settings. The diffusion cache was rebuilt
from the same corpus with only the ranking metric changed, and verified to hold
**2,992 meshes — identical to the Euclidean cache** — so A6 varies one thing.

## G1 restated at 15,000 steps

| | parameters | weight L1 @T=8 | deformation @T=8 |
|---|---|---|---|
| **shared cell, k=4** | **33,665** | **0.0234 ± 0.0007** | **0.00031 ± 0.00001** |
| B3: 8 unshared layers, k=8 | 302,985 (9.0×) | 0.0269 ± 0.0027 | 0.00036 ± 0.00007 |

| condition | measured | threshold | straddle |
|---|---|---|---|
| gain `L1(T=8)/L1(T=1)` | 0.266 ± 0.017 | ≤ 0.80 | no |
| versus B3 @T=8 | 0.878 ± 0.109 | ≤ 1.05 | no |

**GATE G1: PASS**, now on matched arms. The margin narrows with budget — the
shared cell is 12% better here against 19% at 5k — but no seed crosses either
threshold, and it holds at 9.0× fewer parameters.

## A6

Each arm evaluated on the cache it trained on.

| | coverage (probe) | deformation @T=8 | weight L1 @T=8 |
|---|---|---|---|
| Euclidean | mean 0.9984, p5 1.0000 | **0.00031 ± 0.00001** | **0.0234 ± 0.0007** |
| diffusion | mean 0.9320, **p5 0.3106** | 0.00412 ± 0.00003 | 0.0487 ± 0.0011 |

Diffusion-ranked candidates are **13× worse on deformation**, and the result is
conservative in its favour: each arm is scored against the ground truth
projected onto *its own* candidate set, so the diffusion arm is measured against
an easier, worse target than the Euclidean arm is. Its true error against the
artist weights is larger than shown.

## Decisions

**1. Euclidean candidate selection is confirmed. ADR-0006 stands.** What was
decided there on oracle coverage alone now holds after training, at three seeds
and the full budget.

**2. The bleeding comparison is not usable, and is excluded.** Bleeding is mass
on candidates beyond a fixed distance, and each cache stores distances in its
own metric — Euclidean in one, diffusion in the other. The two columns are not
on the same scale, so the apparent 0.1120 versus 0.2123 says nothing. Recorded
here because it is an inviting number to quote.

**3. A6 does not test H1, and the vital-ablation list is wrong about that.**
This is the finding worth keeping. `PAIR_FEATURE_DIM` is 7 and its second
component is **diffusion distance** — so *both* arms already receive geodesic
information as a per-pair feature. A6 varies only which metric ranks the top-8
candidates.

H1 claims that *injected geodesy* lets anatomical separation emerge. A6 as
specified cannot speak to it, because geodesy was never withheld from either
arm. What A6 actually settles is narrower and still worth having: **diffusion
distance is a worse candidate selector than Euclidean distance on D1.**

The severity of the ceiling reinforces that this experiment cannot be stretched
further. At p5 = 0.3106 the hardest 5% of probe vertices can hold under a third
of their ground-truth mass whatever the model does, and a mean displacement
error is exactly the kind of statistic a small number of catastrophic vertices
dominates. Attributing the 13× gap between "geodesy does not help" and "top-8
by diffusion distance is a bad selector" is not possible from this run, and the
second reading alone is sufficient to explain it.

## Consequences

The real H1 ablation is cheap and has not been run: **drop diffusion distance
from the pair features, keep Euclidean ranking.** One arm, 3 seeds, ~3h, and no
cache rebuild — the feature is already computed and merely needs masking. Until
then H1 is untested rather than refuted, which is what ADR-0006 said and what
this ADR continues to say.

The vital four (A1, A2, A4, A6) are now complete as written, with the caveat
that A6 answered a different question than the one it was listed for.
