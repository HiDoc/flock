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
