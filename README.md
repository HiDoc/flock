# FlockRig / FlockSkin

A learned, recurrent, weight-shared **local rule** for mesh skinning — and, in V1,
for joint placement.

## The thesis

Not "a small network can skin": a feed-forward GNN already does that. The claim is
narrower and falsifiable:

> A local recurrent dynamics with shared weights — a learned descendant of the
> flocking rules, in the Graph Neural Cellular Automaton family — converges to an
> **attractor** that is a valid skinning solution, **stays** there, and **repairs**
> it after local corruption.

Three properties follow, and V0 exists to decide them fast:

| | Property | Decided by |
|---|---|---|
| **P1** | T iterations of one cell beat a single pass, and match an unshared GNN ~8x its size | gate G1 — *eliminatory* |
| **P2** | Iterating past convergence does not degrade the result | gate G2 |
| **P3** | Local corruption is repaired, without collateral damage elsewhere | gate G3 |

If P1 fails, this is another GNN and the project pivots or stops. That outcome is
a success of the method, not of the model.

The lineage is deliberate: Reynolds' boids (1987) → cellular automata → Neural CA
→ Graph NCA. Separation, alignment and cohesion each have an analogue here — but
they are *learned* rather than coded, and Reynolds' rules are missing the one this
system needs most, which is how to stop. Hand-coded Reynolds is therefore a
baseline in its own right (B5): if three fixed rules suffice, learning the cell is
not justified.

## Status

| Gate | Question | Threshold | Status |
|---|---|---|---|
| G0 | Do the candidates contain the GT weight mass? | mean ≥ 0.98, p5 ≥ 0.90 | **PASS** — 0.9901 / 0.9647 on D1 (2,992 meshes) |
| G0.5 | Does the loop learn at all? | single-mesh L1 → ~0 in < 10 min | **PASS** — L1 0.554 → 0.031 in 6.1 min |
| G1 | Is the recurrence useful? (P1) | L1(T=8) ≤ 0.8·L1(T=1), within 5% of B3 | not run |
| G2 | Is there an attractor? (P2) | < 5% degradation T=8 → T=32 | not run |
| G3 | Does it repair? (P3) | ≥ 80% resolved, < 10% collateral | not run |
| G4 | Is it useful at all? | beats B1 and B5 on deformation error | not run |

**Milestone: M3 — pooled training.** The state pool, the C0-C2 corruption
curriculum and all three losses are running end to end; `flock eval` traces every
metric by iteration count. G0 and G0.5 have passed; the gates so far also
overturned one design assumption
([ADR-0006](docs/adr/0006-euclidean-candidates-on-d1.md)) and revised the
throughput budget ([ADR-0007](docs/adr/0007-throughput-measured-on-m4.md)).

Next is M4 — baselines B0-B5 and ablations A1/A2/A4/A6 — where **G1 decides P1,
and with it whether the project continues**.

## Layout

```
src/flock/
  domain/       pure value objects and contracts — numpy and stdlib only
    geometry/     Mesh, Skeleton, PoseBank        — what a rig is
    skinning/     WeightField, LBS, corruption    — what a solution is
    dynamics/     CellState, StatePool, ports     — the local rule
    sample.py     PreprocessedMesh                — the aggregate
  data/         D0 generation, the ACL, preprocessing, the .npz store
  backends/mlx/ the cell, rollout, fused LBS, the compiled train step
  training/     losses, corruption curriculum, the pool-driven loop
  evaluation/   metrics by iteration, baselines B0-B5, ablations A1-A9, gates
  experiment/   run naming, config and metric serialisation
  cli/          flock ...
```

`domain/` imports no framework and no I/O. That is checked by
`tests/unit/test_architecture.py`, not by convention.

## Getting started

```bash
uv sync
uv run flock --help
uv run flock gates          # the decision table
uv run pytest

# build D1 and run the gate (needs the corpus in data/raw/articulation-xl2)
uv run flock data preprocess --source d1 --root data/raw/articulation-xl2 --out data/cache/d1
uv run flock oracle coverage --cache data/cache/d1 --source d1
uv run flock train --overfit 0 --change g05-sanity   # gate G0.5
uv run flock train --change my-experiment            # pooled training run
uv run flock eval --run v0_YYYYMMDD_my-experiment    # metrics by iteration
```

## Why the structure looks like this

Two of the spec's risks are silent ones: candidate sets that cannot reach the
ground truth (R1), and ground truth leaking into features or candidates (R7).
Both are data-invariant problems, and both were checklist items. Here they are
constructor invariants instead — `CandidateTable.from_diffusion_distance` takes
no weights of any kind, and `PreprocessedMesh.validate()` runs on every load, so
a stale cache fails at read time rather than as an unexplainable curve three days
later.

That is the whole argument for the domain-driven layering. The ceremony that
usually travels with it — aggregate roots with behaviour over arrays, domain
events, service classes wrapping a single function, any per-operation
abstraction — is explicitly rejected in [ADR-0001](docs/adr/0001-ddd-for-research-code.md).

## Documents

* [docs/spec.md](docs/spec.md) — the research plan (English, canonical)
* [docs/spec.fr.md](docs/spec.fr.md) — the original French document
* [docs/glossary.md](docs/glossary.md) — term ↔ symbol ↔ code ↔ spec section
* [docs/adr/](docs/adr/) — architecture decisions

## Data

| Tier | Source | Meshes | Ground truth |
|---|---|---|---|
| D0 | procedural capsule rigs | 200-500 generated | analytic |
| D1 | Articulation-XL 2.0, canonical subset | 2,992 | artist, one 22-joint skeleton |

D1 is selected out of Articulation-XL 2.0 (MagicArticulate, CVPR 2025) by three
rules in [articulation_xl.py](src/flock/data/acl/articulation_xl.py): Mixamo
naming, a core that reduces exactly to the canonical 22 joints, and at most 20%
of weight mass on fingers. 22.5% of the corpus survives, on **one** hierarchy.
Meshes failing the rules are dropped rather than normalised — see
[ADR-0005](docs/adr/0005-d1-tier-from-articulation-xl.md). D0 still runs first:
its ground truth is analytic, so G0-G3 are settled where any gap is the model's.

## Target

Apple M4, 24 GB unified memory, MLX/Metal. Roughly one hour of useful training
per experiment. Note that the spec sizes its throughput budget against an M4
**Pro**; see [ADR-0003](docs/adr/0003-fixed-shapes-and-npz-schema.md).
