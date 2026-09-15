# ADR-0011 — A4: damped deltas confirmed, and the drift is not accumulation

**Status:** accepted · **Date:** 2026-09-01 · **Evidence:** A4, 3 seeds per arm

## Context

Ablation A4 (spec §4.3, *vital*) asks whether the damped-delta update
`z ← z + α·Δz` with α=0.25 is load-bearing, or whether predicting the logits
outright does as well. `cell.py` states the design claim plainly: absolute
prediction "turns the system into a feed-forward GNN wearing a costume", because
the last iteration overwrites everything before it and `Δz → 0` stops being a
representable fixed point.

A4 also became a test of something else. [ADR-0009](0009-the-drift-is-constant-velocity.md)
found G2's drift to be a constant per-step bias and concluded `L_stab` was the
indicated fix, reasoning that a constant bias *is* a persistent non-zero `Δz`.
An absolute arm has no increment to accumulate, so if the drift were an artifact
of integration it should vanish here.

## Measured

Identical training: 5,000 steps, 400 meshes, lr 1e-3, 33,665 parameters in both
arms — A4 changes the update rule, not the architecture.

| | deltas (design) | absolute (A4) |
|---|---|---|
| weight L1 @T=8 | **0.0321 ± 0.0019** | 0.1667 ± 0.0282 |
| deformation @T=8 | **0.00038 ± 0.00003** | 0.00177 ± 0.00032 |
| per-step stability @T=8 | **0.000203 ± 0.000093** | 0.008560 ± 0.000780 |
| drift T=32/T=8 | **1.223 ± 0.058** | 1.349 ± 0.112 |

Weight L1 by T, averaged over seeds:

| | T=0 | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|---|---|
| deltas | 0.1603 | 0.1173 | 0.0728 | 0.0368 | **0.0321** | 0.0339 | 0.0393 |
| absolute | 0.1603 | **0.7161** | 0.5882 | 0.1280 | 0.1667 | 0.1995 | 0.2228 |

## Decisions

**1. The damped-delta update is confirmed, decisively.** 5.2× better weight L1,
4.7× better deformation, and 42× smaller per-step change at the same parameter
count. The design claim in `cell.py` was not decoration.

**2. The curve shape is the real finding, not the endpoint.** The absolute arm's
first step makes the input **4.5× worse** — 0.1603 → 0.7161 — and it never
recovers to its own T=4 value. That is exactly baseline B3's pathology: only the
converged endpoint is a valid answer, so intermediate states are worse than
doing nothing. The delta arm improves monotonically to T=8 because every state
is a solution. This is the difference between a dynamics and a pipeline, and A4
shows it is the *update rule* that creates it, not depth or parameter count.

**3. G1's gain half can be gamed, and A4 demonstrates it.** The absolute arm
"passes" `L1(T=8)/L1(T=1) = 0.246 ≤ 0.80` — better than the delta arm's 0.274 —
purely because its T=1 is catastrophic. A ratio to a terrible starting point
flatters a terrible model. The gain criterion is only meaningful alongside the
absolute level and the curve shape, which is why G1 requires the B3 comparison
too; recorded here so the ratio is never quoted alone.

**4. The drift is not integration. This corrects ADR-0009's reasoning.** The
absolute arm has no increment to accumulate and drifts **more** (1.349 vs
1.223), monotonically from T=4 onward. So repeated application of the cell moves
the state away from the ground truth as a property of the *mapping itself*, not
because small biased increments pile up.

ADR-0009's decision — that `L_stab` is worth extending — is not refuted, since a
stability term penalises frame-to-frame change whatever its origin. But its
stated reason was wrong, and the mechanism it assumed is now ruled out. Any fix
has to change what the cell outputs at a converged state, not how that output is
integrated.

## Consequences

A4 is settled and needs no repeat. Two cautions carry forward.

The repair and drift-control figures invite a normalisation error. The absolute
arm shows `repaired_c1` 0.586 against the delta arm's 0.505, and a *lower*
drift-control at 0.327 against 0.734 — while being five times worse in absolute
terms. Both metrics are relative to that arm's own settled state, so a worse
model has more headroom on one and a larger denominator on the other. Neither
number says the absolute arm is better at anything, and neither should be quoted
without the absolute error beside it.

A6 (diffusion versus Euclidean candidates) remains the last vital ablation. It
needs the cache rebuilt with the diffusion metric, so it is not an hour's work
like this one.
