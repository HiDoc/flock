# ADR-0002 — A module-level backend port, not an array abstraction

**Status:** accepted · **Date:** 2026-08-31

## Context

Spec §7.3 requires the code to stay portable if a compute wall ever forces GPU
rental: "le code reste portable si le backend est isolé". The obvious reading is
a `Backend` protocol wrapping the array operations we use — `take`, `softmax`,
`einsum` — with an MLX implementation today and a torch one later.

That reading is wrong here, and expensively so.

Spec §3.5 asks for `mx.compile` over the entire training step, fixed shapes
throughout, gather-only access, and exactly one `mx.eval` per optimiser step.
Risk R5 is explicit that throughput at this model size dies of Python and
kernel-launch overhead rather than arithmetic — the tensors are small, so
per-call cost dominates. A per-operation indirection layer sits directly in that
hot path and is precisely the overhead R5 warns about. It would also obscure the
graph the compiler is trying to see.

## Decision

Put the port at **function granularity**, never at operation granularity.

`domain/dynamics/ports.py` defines four calls: `init_params`, `make_bank`,
`make_pool_step`, `trace_weights`. An adapter implements them however its
framework prefers. `backends/mlx/` is written in raw, idiomatic `mx` — no
wrappers, no shims, nothing of ours between the code and the compiler.

**Everything crossing the interface is NumPy or opaque.** The device exists only
on the far side: `training/` and `evaluation/` hold parameter trees they never
read and arrays that are always `ndarray`.

The genuinely framework-neutral seam is not the protocol. It is **the `.npz` file
on disk**: fixed-shape numpy arrays that any framework can read. Preprocessing
produces them, every backend consumes them.

A second backend is therefore a sibling package (`backends/torch/`) that reads
the same files and implements the same four calls, not a set of operations
plugged into a shared skeleton.

### Amendment, 2026-08-31: the first version of this port was decorative

As originally written, the port exposed `rollout`, `lbs` and a `train_step`
taking device arrays. That drew the boundary in the wrong place. The training
loop still constructed `mx.array`s, called `mx.eval`, and imported
`flock.backends.mlx.cell` directly; `get_backend()` was never called from
anywhere, and `TrainConfig.backend` was never read. The protocol existed, was
documented, and carried no weight — precisely the abstraction-with-one-caller
that [ADR-0001](0001-ddd-for-research-code.md) rejects.

The layering test did not catch it, because it enforces *which layers may
import which* and `training → backends` is legitimate. What it could not see was
that the import named a **concrete adapter** rather than the port.

Two changes fixed it. The boundary moved up so that a step takes and returns
NumPy, which removed every `mx` reference from `training/` and `evaluation/`.
And `MeshBank` moved from `training/` to `backends/mlx/bank.py`, because device
residency is exactly the sort of thing a second backend implements differently —
unified memory holds the tier outright, discrete memory would stage and stream.

The layering test now names `mlx` and `flock.backends.mlx` as forbidden imports
for both layers, so the regression is caught by CI rather than by inspection.

## Consequences

Some structural duplication between backends is accepted — the rollout loop would
be written twice. That is the correct trade: the duplicated code is about thirty
lines, and the alternative taxes every training step of the project's life for a
portability need that may never arrive.

The port stays honest only if it is not widened. Any new call added to `Backend`
must be something a *second* backend would plausibly implement differently. If it
would be identical in both, it belongs in the domain as a plain function.

It also stays honest only if it is *used*. An unused port is worse than none: it
advertises a portability that has never been exercised. The layering test is what
keeps that claim true.

One cost is recorded rather than hidden. Unifying on a single compiled step means
the G0.5 sanity gate now runs the full loss graph with `L_deform` and `L_stab`
weighted to zero, plus the overflow rollout step — around 25% slower than the
dedicated M2 path it replaced (10.9 → 8.4 steps/s). One code path that the real
runs exercise is worth more than a fast gate on a path nothing else uses.
