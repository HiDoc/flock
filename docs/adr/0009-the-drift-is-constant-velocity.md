# ADR-0009 — The G2 drift has constant velocity, and deformation is not blind to it

**Status:** accepted · **Date:** 2026-09-01 · **Evidence:** 14-point T grid, 3 seeds

## Context

G2 fails: `L1(T=32)/L1(T=8) = 1.223 ± 0.058`, no seed passing. Two readings were
on the table, and they call for opposite fixes.

* **Accumulated noise** — each step makes a locally-motivated but globally-noisy
  edit, and the errors pile up. Excess error would grow diffusively, `~√T`,
  decelerating per step. `L_stab` is the right medicine: shrink the step.
* **A growing mode** — the fixed point is unstable and the state accelerates
  away. Excess error would grow faster than linearly. Damping the step size
  delays divergence without removing it, and `L_stab` is the wrong medicine.

The distinction had been argued from **two intervals** (T=8→16 and T=16→32),
which gave an apparent exponent of 1.98 and pointed at the growing mode. Two
points cannot carry that conclusion: the first interval straddles the curve's
minimum, where it is still flattening out, so its rate is understated and the
second interval looks like acceleration. The fit was an artefact of the grid.

## Measured

Fourteen values of T from 4 to 128, on the three shared-cell seeds, no retraining.

| | |
|---|---|
| minimum | T = 8 (0.03213) |
| power-law fit of the excess | `(T−8)^**1.15**`, R² = 0.990 |
| per-step growth, T=12 → T=128 | flat at ~0.00031 L1/step |
| per-step state change (median) | flat at ~0.00016 L1/step |

**Constant velocity.** Not diffusive (0.5), not accelerating. The state moves in
a consistent direction at a fixed rate and shows no sign of stopping or of
running away, over sixteen times the trained horizon.

## Decisions

**1. `L_stab` is the right target after all.** *(Superseded in part by [ADR-0011](0011-a4-damped-deltas-confirmed.md): A4's absolute arm has no increment to accumulate and drifts more, so the mechanism assumed below is ruled out. A stability term may still suppress the drift, but not for this reason.)* A constant per-step bias is
precisely a persistent non-zero `Δz`, which is what a stability term penalises.
The growing-mode worry is retired. This reverses an intermediate call made from
the two-point fit, and is the reason the dense grid was run before spending an
hour of training on the fix.

**2. Report stability per step, not per checkpoint.** `EVAL_ITERATIONS` is
deliberately non-uniform, and `stability` was being measured between consecutive
*checkpoints*. Over a widening grid a constant rate therefore reads as
divergence — the raw column climbed 0.00038 → 0.0057 while the true rate never
moved. The metric is now `stability_per_step`, divided by the gap. Renamed
rather than silently corrected, so old curves cannot be compared against new
ones by accident.

**3. Deformation fails G2 too. The "blind reference metric" claim is withdrawn.**

| metric | T=8 | T=32 | ratio | needs |
|---|---|---|---|---|
| weight L1 | 0.03213 | 0.03932 | 1.223 ± 0.058 | ≤ 1.05 |
| deformation | 0.00038 | 0.00044 | **1.153 ± 0.043** | ≤ 1.05 |

No seed passes on either. Deformation had been reported as "flat at 0.0004 from
T=4 to T=32" — an artefact of rounding to four decimals, which is where the
whole curve lives. At T=128 it reaches 0.00070, nearly double the minimum.

The same rounding cut the other way in G1, where both arms' deformation at T=8
was reported as "0.0004". At five decimals the shared cell scores 0.00038 ±
0.00003 against B3's 0.00045 ± 0.00002 — 16% better on the reference metric, a
stronger claim than the one that was published. Four decimals is simply the
wrong precision for a metric whose whole range lives in the fourth.

This removes the consoling reading. It was tempting to treat the drift as
motion along directions the reference metric cannot see — the H3 story, where
artist weights are non-unique and L1 punishes correct solutions. That story is
wrong here: the rig genuinely degrades, just more slowly than L1 suggests
(15% against 22%). H3 is not refuted in general; it simply does not excuse
this.

## Consequences

*Superseded by [ADR-0015](0015-the-dynamics-is-a-flow-not-an-attractor.md), which measures the character of the motion directly: a smooth near-one-dimensional monotone flow, with error growing 53x from a ground-truth start over 1,024 steps. The constant-velocity description below holds only over the window it was fitted on.*

G2's failure is specific rather than mysterious: **the dynamics has a constant
non-zero velocity over T=12–128 and never quiesces.** Past T=128 the rate bends
upward (0.00032/step at T=32→64 against 0.00047/step at T=128→256), so the
constant-rate description is scoped to the measured range.

**Where it goes.** Out to T=256 the distribution's shape is nearly invariant —
max weight 0.832 → 0.826, entropy 0.380 → 0.393, L1 to uniform 1.570 → 1.564 —
while L1 to ground truth grows 680-fold. The drift is not diffusion toward
uniform and not collapse onto one bone: mass migrates between candidates at
roughly constant concentration. Vertices change allegiance.

**On the Reynolds reading.** It is tempting to call this Reynolds' missing
fourth rule (spec §0.2) — boids have no quiescent behaviour, and neither does
this cell. That framing is *compatible* with every number here and *supported*
by none of them. The cell does not run Reynolds' rules, and a much cheaper
explanation is available: it is trained at k=4 with one overflow step, so
nothing has ever asked it to hold still at T=128, and linear extrapolation off
the training manifold requires no theory at all. The mundane reading is the one
to rule out first.

The framing would earn its place if hand-coded B5, at a non-degenerate
configuration, showed the same signature — constant-rate migration at fixed
concentration. That has not been run. Until it is, the spec's §0.2 remark stands
as a well-placed prior, not as a result.

Two experiments follow, and they are now cleanly separated:

* **Extend `L_stab`** — more overflow steps, higher weight. Directly targets a
  constant bias. Predicted effect: the per-step rate falls; if it merely gets
  smaller without reaching zero, the term is buying time rather than a fixed
  point.
* **A7, stochastic update (fire rate 0.5)** — targets the C2 local-optimum
  failure of [ADR-0008](0008-repair-protocol-and-the-drift-control.md), and may
  also break a coherent constant-velocity mode that synchronous updates sustain.

The G3 collateral finding is unchanged and now better explained: of 0.768 raw
collateral, 0.734 was this drift, at a rate the dense grid confirms is constant.
