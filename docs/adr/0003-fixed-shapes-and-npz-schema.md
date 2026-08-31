# ADR-0003 — Fixed shapes, the `.npz` schema, and the hardware caveat

**Status:** accepted · **Date:** 2026-08-31

## Context

`mx.compile` recompiles whenever an input shape changes. A single varying
dimension anywhere in the training step silently converts every step into a
compilation, and the symptom is indistinguishable from "the model is just slow".

Spec §3.5 therefore requires fixed shapes throughout, which pushes the constraint
all the way back into the data layout.

## Decision

Freeze three constants and pad everything to them:

| Constant | Value | Meaning |
|---|---|---|
| `NUM_VERTICES` | 1024 | vertices per mesh, padded with an explicit mask |
| `NUM_NEIGHBOURS` | 8 | one-ring, topped up by KNN below valence 8 |
| `NUM_CANDIDATES` | 8 | candidate bones per vertex |

Every array in the `.npz` is shaped from these. Masks are explicit and stored,
never inferred from sentinel values — a padded vertex must be identifiable
without a convention about zeros.

Access is **gather-only**: `[V, K]` and `[V, B]` are constant index tables, so
aggregation is a `take` plus a reduction, which is the pattern Metal handles
best. No scatter anywhere.

`SCHEMA_VERSION` is stored in every file and checked on load. When the layout
changes, old caches fail loudly instead of being silently misread.

## The hardware caveat

The spec targets a **MacBook Pro M4 Pro, 24 GB** and sizes its budget against it:
5-15 optimiser steps/s, 10-15k steps in 40-50 minutes (§3.4).

**This machine is an Apple M4 (base), 24 GB** — roughly half the GPU cores of an
M4 Pro.

Memory is unaffected and remains a non-issue: dataset, pool and model together
stay under 1 GB against 24 GB of unified memory, so the whole dataset lives
resident as arrays with no loader and no host-device copy.

Throughput is affected, and the spec's figure must be **measured at gate G0.5**
rather than inherited. It is recorded in the run metrics from that point on. Note
the diagnostic order this implies: if the step rate disappoints, the first
hypothesis is a stray varying shape causing recompilation (R5), not the GPU. Rule
that out before touching anything else, and before concluding the hardware is the
constraint.

## Consequences

Meshes above 1024 vertices are decimated; below, they are padded. V1's scale-up
to 4-8k vertices (spec O2) changes these constants, which is exactly why they are
named constants with a schema version behind them rather than literals.
