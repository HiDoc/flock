# ADR-0006 — Candidates on D1 are ranked by Euclidean distance, not diffusion

**Status:** accepted · **Date:** 2026-08-31 · **Evidence:** gate G0 over 2,992 meshes

## Context

Hypothesis **H1** is the spec's sharpest correction to the original brief. Local
rules see Euclidean distance, so two regions close in space but far across the
surface — thigh against thigh, arm against torso — produce structural bleeding
that no amount of local iteration can undo. The remedy §1.1 prescribes is to
inject geodesy offline and **select candidates on diffusion distance**.

Gate **G0** exists to test that prescription before any training: if the
ground-truth weight mass does not live inside the B=8 candidates, there is a
ceiling the model cannot cross (R1), and every later curve inherits it.

## What G0 measured

Two obstacles surfaced first, both properties of the corpus rather than of H1:

1. **Split-seam soup.** Source meshes carry 3–51% exact duplicate vertices from
   FBX export, shattering one character into up to 2,835 nominal components. Heat
   cannot cross a gap, so geodesic distance was meaningless. Welding coincident
   vertices brought a representative mesh from 2,835 components to 21, and most
   of the tier to 1.
2. **QEM is unusable here.** `igl.decimate` assumes manifold input; on these
   meshes it returned 8,506 vertices for a 2,838-face budget on one model and
   *zero* on two others. Replaced by vertex clustering (ADR-0005 addendum in
   `decimate.py`), which cannot fail and hits the budget by construction.

With both fixed, and diffusion distance made component-aware — seeded per
component along each bone's length, with the Euclidean gap added back — the two
candidate metrics were measured head to head on the same 30 meshes:

| Candidate ranking | coverage mean | coverage p5 | meshes passing |
|---|---|---|---|
| diffusion | 0.928 | 0.514 | 3 / 30 |
| **Euclidean** | **0.992** | **0.930** | **28 / 30** |

G0 thresholds are mean ≥ 0.98 and p5 ≥ 0.90. Diffusion selection fails; Euclidean
passes. Over the full tier of 2,992 meshes, Euclidean scores **0.9901 / 0.9647**,
with 2,649 meshes passing individually.

## Decision

**On D1, candidates are ranked by Euclidean point-segment distance.** The metric
is a config field (`PreprocessConfig.candidate_metric`), so ablation **A6** flips
it rather than requiring a code change.

**Diffusion distance is retained as a pair feature regardless.** It sits in
`pair_features` alongside the Euclidean distance, so the anatomical signal H1
argues for is still available to the cell. What changed is the *ranking*, not the
information: a bone that Euclidean proximity admits as a candidate can still be
suppressed by the learned dynamics on diffusion evidence — which is, in flocking
terms, exactly where separation ought to operate.

## What this does and does not say about H1

It does **not** refute H1. Three readings remain open, and G0 cannot separate
them:

* Mixamo's own auto-rigging is largely volumetric, so its ground truth may
  correlate with Euclidean proximity by construction — in which case D1 is a
  biased instrument for this particular question, and D0's analytic ground truth
  is the fair test.
* Our diffusion distance is computed on ~900-vertex clustered meshes with
  residual non-manifold structure; the metric may be too noisy at that
  resolution to rank 22 bones reliably.
* H1 may simply be wrong about *selection* while remaining right about
  *representation* — which is what keeping the feature hedges against.

What G0 establishes is narrower and sufficient to act on: **with the diffusion
distance we can actually compute on this corpus, candidate selection by diffusion
would have trained under a ceiling of 0.93 coverage and a 5th percentile of
0.51.** Spec §4.4 is unambiguous about what to do with that — fix the candidates,
do not train. This is the gate performing its function, three days before it
would otherwise have shown up as an unexplained loss floor.

A6 is now a stronger experiment than planned: it has a measured baseline on both
arms rather than an assumption on one.
