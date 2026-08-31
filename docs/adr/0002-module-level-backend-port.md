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

`domain/dynamics/ports.py` defines four calls: `init_params`, `rollout`, `lbs`,
`train_step`. An adapter implements them however its framework prefers.
`backends/mlx/` is written in raw, idiomatic `mx` — no wrappers, no shims,
nothing of ours between the code and the compiler.

The genuinely framework-neutral seam is not the protocol. It is **the `.npz` file
on disk**: fixed-shape numpy arrays that any framework can read. Preprocessing
produces them, every backend consumes them.

A second backend is therefore a sibling package (`backends/torch/`) that reads
the same files and implements the same four calls, not a set of operations
plugged into a shared skeleton.

## Consequences

Some structural duplication between backends is accepted — the rollout loop would
be written twice. That is the correct trade: the duplicated code is about thirty
lines, and the alternative taxes every training step of the project's life for a
portability need that may never arrive.

The port stays honest only if it is not widened. Any new call added to `Backend`
must be something a *second* backend would plausibly implement differently. If it
would be identical in both, it belongs in the domain as a plain function.
