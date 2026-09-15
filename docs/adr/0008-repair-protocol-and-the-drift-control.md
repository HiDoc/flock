# ADR-0008 — The repair protocol, and why collateral damage needs a drift control

**Status:** accepted · **Date:** 2026-09-01 · **Evidence:** gate G3, 3 seeds

## Context

P3 claims the dynamics *repairs* local corruption. Gate G3 quantifies it: at
least 80% of the induced error in the region Ω resolved within 8 steps, with
under 10% relative collateral damage outside it.

The C1 evaluation already run at every checkpoint does **not** test this. There,
the ground truth is corrupted and the cell solves from a cold start with a zero
hidden state — a harder initialisation, but a different claim. P3 is about a
state that has already converged and is then damaged, which is also exactly what
the state pool's recorruption cycle (§3.1) trains for.

## Decisions

**1. Damage a converged state, not the ground truth.** The protocol settles for
8 steps from the GT initialisation, corrupts the *model's own* settled state,
and resumes. This required exposing the hidden state across the backend port
(`trace_from`, returning `RolloutFrame`), because the dynamics cannot be stopped
and resumed through a weights-only interface.

**2. Zero the hidden state inside Ω.** A repair that ran with the hidden state
intact would be the cell reading its own memory of the answer rather than
recovering it from its neighbours. That is not the property under test. Outside
Ω the hidden state is kept: the surrounding tissue is undamaged by construction.

**3. Report collateral damage against an undamaged control.** This is the
decision that changed a conclusion. Raw collateral at T=8 is **0.768 ± 0.143**,
a flat failure of the 0.10 threshold. But the same state, run the same number of
further steps with **no damage at all**, drifts by **0.734 ± 0.154**. Only
**+0.034** is attributable to the damage — which passes.

Charging that drift to the damage would have produced a false finding: "repair
smears error into the correct region", the R3 do-no-harm failure. What is
actually happening is the G2 drift, already known and reported, measured on a
smaller baseline. `evaluate_repair` therefore reports `drift_relative` beside
`collateral_relative` always, not on request.

**4. Report how far the state moved inside Ω.** `repaired_fraction ≈ 0` has two
completely different causes — the cell tried and failed, or the cell saw nothing
to fix — and the gate metric cannot distinguish them. `moved_inside` can, and on
C2 it is what carries the finding.

## Consequences

G3 **fails**, and the two corruption levels fail for different reasons.

| | induced L1 | moved inside Ω @T=8 | repaired @T=8 |
|---|---|---|---|
| C1 random patch | 1.606 | 1.115 ± 0.233 | 0.505 ± 0.169 |
| C2 hierarchy transfer | 1.695 | **0.007 ± 0.004** | **−0.001 ± 0.001** |

C1 repair is real but **saturates**: 0.505 at T=8, 0.510 at T=16, 0.507 at T=32.
It is not slow, it is stuck — half the induced error is permanent, and four
times the budget buys nothing.

C2, at the same damage magnitude, does not move at all.

**Why, established by a 2x2 probe** (3 seeds, same settle-damage-resume protocol).
The first reading recorded here was that a C2 patch is *locally self-consistent*
— every vertex agreeing with its neighbours, leaving a purely local rule no
signal. Measurement refutes it. A C2 patch is not coherent: 4.8 distinct target
bones per patch, and 41% of neighbour pairs inside it disagree. Two synthetic
probes then separate the candidate causes:

| damage | entropy in patch | neighbour agreement | moved inside | repaired |
|---|---|---|---|---|
| C1 random patch | 1.891 | 12% | 1.115 | 0.505 |
| C2 hierarchy transfer | 0.164 | 48% | 0.007 | −0.001 |
| **peaked-random** — wrong, inconsistent, but confident | 0.000 | 13% | **0.000** | 0.000 |
| **flat-hierarchy** — C2's bone error, spread out | 1.920 | 60% | **0.883** | 0.441 |

Neighbour agreement does not predict repair; it is mildly *anti*-correlated.
**Entropy predicts it completely.** The cell engages with a flat state and
ignores a peaked one, whether or not the peaked one is correct and whether or
not its neighbours agree.

The mechanism is a learned shortcut: the cell detects *"this looks unsolved"*
rather than *"this is wrong"*. Every corruption it was trained to fix arrives
flat — C0 adds noise, C1 writes uniform-random rows — while every state it was
trained to leave alone (the CLEAN fraction, R3's do-no-harm objective) is
peaked. Peakedness is therefore a near-perfect proxy for "needs work" across the
curriculum, and gradient descent found it. **A confidently wrong state is
invisible to the model.**

**Coverage is not the explanation, though it first looks like one.** C2 enters
`default_curriculum` only at step 8,000 and every run here is 5,000 steps, so
these models never trained on C2 at all — a fact worth stating plainly, since an
earlier draft of this ADR claimed the opposite. But absence from training cannot
be the cause: `flat-hierarchy` and `peaked-random` were equally absent, and one
is repaired at 0.441 while the other does not move at all. What separates them
is entropy, not familiarity. The shortcut generalises: the model applies "flat
means unsolved" to corruptions it has never seen, and is blind to peaked ones it
has never seen. Detecting C2 requires evaluating correctness, which is strictly
harder than detecting flatness, and while the easy signal covers the training
distribution there is no pressure to learn the hard one.

This bounds P3 much more sharply than the flocking reading did, and the bound is
about the training signal rather than about locality. It also predicts where a
fix has to act: a curriculum whose corruptions are peaked-but-wrong, so that
confidence stops being evidence of correctness. Whether the architecture can
learn that at all is open — ablation A5 (bone channel) is the natural place to
look for the extra information it would need.

The collateral half of G3 is **not** a do-no-harm failure. The drift is, once
again, G2. Two gates now fail for one cause, which concentrates the next
experiment rather than dividing it.
