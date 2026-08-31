# ADR-0001 — Domain-driven structure for research code, and what we reject

**Status:** accepted · **Date:** 2026-08-31

## Context

Applying enterprise DDD wholesale to numerical research code is a well-known way
to produce something slow, unreadable and hostile to experiment. Array code wants
to be flat functions over tensors; wrapping tensors in behaviour-rich objects
fights both the hardware and the reader.

But this project has a specific problem that the DDD toolkit happens to solve.
Two of the spec's seven risks are *silent* ones:

* **R1 — candidate coverage.** If the ground-truth weight mass lives outside the
  B candidate bones, there is a performance ceiling no amount of training can
  cross. The symptom is a loss that plateaus at a value nobody can explain.
* **R7 — ground-truth leakage.** If GT weights influence feature or candidate
  computation, results look excellent and reproduce nowhere.

Both are data-invariant problems. Both were mitigated in the spec by a checklist.
Checklists are the first thing to erode under deadline, and both failure modes
are invisible for days — they surface as a curve, and the curve is what the gates
read.

## Decision

Adopt the parts of DDD that address this, and no more.

**Adopted:**

* **Bounded contexts** — `geometry`, `skinning`, `dynamics`, `data`,
  `training`/`evaluation`. Drawn along the spec's own seams.
* **Ubiquitous language** — the spec's vocabulary is the code's vocabulary, one
  to one, indexed in [glossary.md](../glossary.md). A reader of §4.1 can find
  `bleeding_mass` without a translation step.
* **Value objects with constructor invariants** — the actual payoff.
  `CandidateTable.from_diffusion_distance` accepts distances and a mask; it has
  no parameter that could carry ground truth, so R7 becomes a signature rather
  than a rule. `PreprocessedMesh.validate()` runs on every load, so a stale cache
  fails at read time.
* **Ports and adapters** — see ADR-0002.
* **Anti-corruption layer** — Mixamo and RigNet conventions are translated once,
  at the boundary. Adding the V1 benchmark dataset costs one adapter.
* **Repositories** — for the `.npz` dataset cache and for run directories. Both
  are genuinely collections of persistent things.

**Rejected, explicitly:**

* Aggregate roots with behaviour over arrays. `PreprocessedMesh` validates and
  holds; it does not compute.
* Domain events, CQRS, dependency-injection containers.
* Service classes wrapping a single function. A function is a function.
* Any per-operation abstraction over array ops (see ADR-0002 for why this one
  would actively cost performance).
* Interfaces with one implementation and no second in prospect.

## Consequences

The dependency rule is enforced by `tests/unit/test_architecture.py`, which walks
the AST of every module. `domain/` may import numpy and stdlib, nothing else. A
layering convention that lives only in a README is a convention until the first
deadline; this one fails the build.

The cost is real: value objects add construction-time validation on the data
path. That cost is paid once per sample at preprocessing and once per load, never
inside the training loop, where `CellState` holds raw backend arrays precisely so
that nothing of ours sits in the hot path.

The standing test for any future addition: does it reduce duplication, improve
clarity, or enable a controlled extension that is actually planned? If not, it
does not go in.
