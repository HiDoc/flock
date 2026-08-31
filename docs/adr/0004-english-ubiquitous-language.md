# ADR-0004 — English as the ubiquitous language

**Status:** accepted · **Date:** 2026-08-31

## Context

The research spec is written in French. Its technical vocabulary, however, is
already English: *cell*, *pool*, *rollout*, *bleeding*, *coverage*, *support*,
*gate*, *fire rate*. The French text surrounds English terms of art.

DDD depends on one language shared by the document and the code. Two vocabularies
means a translation step at every reading, and translation steps are where
meaning quietly drifts.

## Decision

**Code, identifiers, docstrings, CLI, tests and ADRs are English.**

`docs/spec.md` is the canonical English translation of the research plan.
`docs/spec.fr.md` keeps the French original — it is the document of record for
the reasoning, and nothing about the translation supersedes it.

Mathematical notation is preserved as written: σ, α, Δz, ‖·‖, §. These are not
prose and transliterating them into `sigma`, `alpha`, `delta_z` would break the
mapping to the spec that this decision exists to protect. Ruff's ambiguous-unicode
rules (RUF001-003) are disabled for that reason, with the reason recorded in
`pyproject.toml`.

## Consequences

The eventual technical report and any publication draw on `docs/spec.md`
directly, with no translation pass at the point where accuracy matters most.

[glossary.md](../glossary.md) is the bridge: every term used in an identifier
appears there with its symbol, its module, and its spec section. Keeping it
current is part of adding a term, not a documentation chore that follows later —
a term absent from the glossary should not appear in an identifier.
