# ADR-0007 — Measured throughput, and the batch size that follows

**Status:** accepted · **Date:** 2026-08-31 · **Evidence:** gate G0.5

## Context

Spec §3.4 budgets 10–15k optimiser steps in 40–50 minutes, reasoning that the
model is ~0.2 TFLOP per step and that even a pessimistic 1 TFLOPS effective
would hold 5–15 steps/s. [ADR-0003](0003-fixed-shapes-and-npz-schema.md) flagged
that this was sized against an **M4 Pro** while the development machine is an
**M4 base**, and deferred the number to G0.5 rather than inheriting it.

Risk **R5** predicts that throughput at this model size dies of Python and
kernel-launch overhead rather than arithmetic. §3.5 gives the diagnostic order
plainly: under 5 steps/s, hunt a silent recompilation before optimising anything.

## Measured

Cell of **33,665 parameters** (H=32, width 96), k=4 rollout with backward pass.

| | steps/s | ms/step |
|---|---|---|
| uncompiled, batch 32 | 1.55 | 645 |
| **`mx.compile`, batch 32** | **2.74** | 365 |

Compilation buys 1.77×. That is below the spec's 5–15, so the R5 question had to
be settled rather than assumed. Sweeping the batch size answers it:

| batch | steps/s | **mesh-steps/s** |
|---|---|---|
| 1 | 77.28 | 77.3 |
| 4 | 17.23 | 68.9 |
| 8 | 10.44 | 83.6 |
| 16 | 5.25 | 84.0 |
| 32 | 2.69 | 86.2 |

**Mesh-steps per second is flat at ~77–86 while steps/s scales inversely with
batch.** That is the signature of a saturated GPU, not of launch overhead: if
Python were the bottleneck, small batches would cost the same per step as large
ones and the right column would climb. This is **not R5**. It is real arithmetic
on a GPU with roughly half the cores the spec assumed, and there is no silent
recompilation to find.

## Decision

**Record ~85 mesh-steps/s as this machine's figure** and size experiments from
it rather than from §3.4.

**Use batch 16 rather than 32 for full training runs.** Since mesh-steps/s is
constant, batch 16 at 5.25 steps/s performs the *same* total work as batch 32 at
2.69 — but it lands 14,175 optimiser steps in 45 minutes against 7,263, which
puts the spec's 10–15k step budget back in reach at identical compute. More
optimiser steps over the same number of mesh-steps is the better trade here:
this is a small model whose regime is set by the state pool's turnover (§3.1),
not by gradient noise at batch 32.

**Do not switch to bf16 yet.** §3.4 makes it conditional on profiling showing a
throughput wall. A wall would be a floor we cannot cross by scheduling; what we
have is a machine that is simply smaller, and fp32 keeps one variable out of
gates G1–G3. Revisit if the ablation matrix stops fitting in a working day.

## Consequences

`configs/base.yaml` moves to batch 16, with the reasoning recorded there.

Gate **G0.5 passed** at these numbers: weight L1 fell from 0.5543 to 0.0308 —
5.6% of its starting value — over 4,000 steps in 369 seconds, comfortably inside
the ten-minute budget §6 allows. The loop learns and the gradients flow.

One caveat carried forward: G0.5 is a sanity check on the machinery, not a
result. It overfits a single mesh under C0 corruption with no state pool and no
deformation loss. Nothing about P1, P2 or P3 follows from it.

---

## Addendum, 2026-08-31: where the training step actually spends its time

The M3 step measured 253.5 ms at batch 16. Profiled by component:

| component | time | share |
|---|---|---|
| **full pooled step** | **253.5 ms** | 100% |
| compiled step, k=4 + overflow | 243.2 ms | 96% |
| compiled step, k=1 + overflow | 92.4 ms | — |
| one LBS (2 poses) | 1.6 ms | 0.6% |
| `bank.gather` | 0.5 ms | 0.2% |
| pool host↔device round trip | 0.2 ms | 0.1% |

The step is ~5 cell applications at ~48 ms each. **Everything outside the
rollout is under 2% combined**, which settles two tempting optimisations before
they are written:

* **Memoising the reference LBS.** `L_deform` poses the ground truth every step,
  and for a given (mesh, pose) that result is constant — a textbook memoisation.
  It is worth **0.6%**. Caching it would mean holding `[meshes, poses, V, 3]`
  and invalidating it whenever the pose bank is resampled, to buy nothing.
* **Keeping the state pool on device.** The host round trip is 0.2 ms. Avoiding
  it would require a scatter in the hot path, which §3.5 forbids, in exchange
  for 0.1%.

Neither is a real optimisation. The only lever that matters is the cell.

### What was done

`concat([a, b, c]) @ W` equals `a @ W_a + b @ W_b + c @ W_c`, so an operand that
is constant along the gathered axis can be projected *before* it is broadcast.
`h_v` enters three MLPs broadcast over K=8 or B=8; projecting it once on
`[N, V, H]` rather than materialising `[N, V, 8, H]` removes about 37% of the
cell's multiply-accumulates.

Measured gain: **1.11×** (253.5 → 229 ms). The arithmetic saving is real and the
result is numerically identical — verified against the concat form to 7e-07 over
three iterations, and the trained checkpoint reproduces its evaluation curves to
1e-07. But a 37% MAC reduction buying 11% says the matmuls were never the
constraint.

### What the constraint is

Forward alone is 61.5 ms against 217.5 ms for forward-and-backward, and a single
`[N, V, K, 96]` activation is **50 MB** in fp32 — roughly six of them per cell
application, all retained for the backward pass. Against ~120 GB/s of memory
bandwidth, this is **bandwidth-bound, not FLOP-bound**. That also explains why
ADR-0007's original finding (mesh-steps/s flat across batch size) held so
cleanly: the machine saturates on traffic, not arithmetic.

The confirmation is bf16, which halves that traffic:

| precision | forward | forward + backward |
|---|---|---|
| fp32 | 61.5 ms | 217.5 ms |
| **bfloat16** | **35.9 ms** | **125.3 ms** |

**1.74×** — close to the 2× a purely bandwidth-bound kernel would give.

### Why bf16 is measured but not adopted

§3.4 permits bf16 "only if profiling shows a throughput wall", and this profile
is that wall. It is still not switched on, for two reasons that are about the
science rather than the speed:

1. **It would need fp32 master weights.** Pure bf16 has ~3 decimal digits of
   mantissa; running Adam's updates in it silently degrades training rather than
   failing. That is a real implementation, not a flag.
2. **It could corrupt the measurements P2 depends on.** The stability signal
   sits at 1e-3 and falling (`L_stab` reached 0.0008), traced over 32 iterations
   of an accumulating `Δz`. bf16 rounding is of the same order. Adopting it
   before G2 risks answering "is there an attractor?" with a rounding artefact.

The precondition for adoption is therefore an fp32-versus-bf16 comparison of the
by-iteration curves showing the T=8→T=32 behaviour unchanged — not a benchmark.
Until then fp32 keeps one variable out of gates G1–G3, as originally decided.
