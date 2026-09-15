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
| G1 | Is the recurrence useful? (P1) | L1(T=8) ≤ 0.8·L1(T=1), within 5% of B3 | **PASS** — restated at 15k on matched arms |
| G2 | Is there an attractor? (P2) | < 5% degradation T=8 → T=32 | **FAIL** — there is no attractor: a monotone flow, error ×53 from a GT start |
| G3 | Does it repair? (P3) | ≥ 80% resolved, < 10% collateral | **not decided** — C1 0.807 (seeds straddle), C2 0.660, C1P 0.764 |
| G4 | Is it useful at all? | beats B1 and B5 on deformation error, from C3 | **PASS** — 3.78× B1, 3.35× B5 (deformation only) |

## Where this sits — and what it is not

Two recent systems already do *template-free rigging of arbitrary assets*, which
is the obvious product ambition for this space:

| | [RigAnything](https://arxiv.org/abs/2502.09615) (TOG 2025) | [SkinTokens / TokenRig](https://arxiv.org/abs/2602.04805) (2026) | **FlockSkin** |
|---|---|---|---|
| scope | skeleton **and** skinning | skeleton **and** skinning, unified | **skinning only** — skeleton is an input |
| method | autoregressive transformer + diffusion | FSQ-CVAE discrete skin tokens, autoregressive + RL | recurrent shared-weight local cell |
| topologies | humanoids, quadrupeds, marine creatures, insects | out-of-distribution assets via RL rewards | **one 22-joint biped skeleton** |
| parameters | large transformer | large transformer | **33,665** |
| from scratch | yes, seconds per shape | yes | yes, but partially — G4 passes on deformation only |
| headline | SOTA across diverse categories | 98–133% skinning accuracy over SOTA | G1 passed; G2/G3 open |

**Read the last row first.** FlockSkin now *does* produce a field from geometry
alone — gate [G4](docs/adr/0013-g4-passes-from-a-cold-start.md) passes, beating
the geometric heuristic 3.78× and tuned Reynolds 3.35× on deformation. But it
improves a bad guess rather than solving the problem: weight L1 falls only
1.2477 → 0.6017, against the 0.0234 the same model reaches when it starts near
the answer. A head-to-head against either system above is still not meaningful —
different data, different metrics, and no skeleton generation here at all.

So this project is **not** a bid to rig anything better. That problem has
credible answers already, trained on the same Objaverse lineage this repo draws
D1 from.

What it is instead is a question those systems do not ask: whether a valid
skinning field is the **attractor of a local recurrent rule** — reached by
iterating one small shared cell, *stable* once reached, and *self-repairing*
after local damage. RigAnything and TokenRig produce a rig in a single forward
or autoregressive pass; neither claims its output is a fixed point that recovers
from corruption. P2 and P3 are properties of a dynamics, not of a map, and they
are what this repo is built to falsify.

That framing is what makes the failures here informative rather than merely
bad. A 33,665-parameter cell losing to a large transformer on accuracy would
say nothing. The same cell failing to hold its own converged state (G2) or to
notice a confidently wrong region (G3) says something specific about local
recurrent dynamics — and those are the results this repo actually has.

One convergent signal worth noting: SkinTokens attributes its gains to
exploiting the **intrinsic sparsity** of skinning matrices. That is the same
structure H5 points at here, and the reason support and bleeding are tracked as
first-class metrics rather than accuracy alone.

**Milestone: M4 — baselines, gates, and the vital ablations (complete).**

> **All results below the G1 section were re-measured at the spec's 15,000-step
> budget** ([ADR-0012](docs/adr/0012-the-budget-was-the-confound.md)). The
> earlier 5,000-step runs silently omitted a whole curriculum stage — C2 enters
> at step 8,000 — and turned a working intervention into a null. G1 alone still
> rests on matched 5k arms and is unaffected.

### The budget was the confound

| | base 5k | base 15k | ce 15k (+C1P) |
|---|---|---|---|
| weight L1 @T=8 | 0.0321 | **0.0234** | 0.0265 |
| drift L1 T=32/T=8 | 1.223 | **1.097** | 1.112 |
| drift deformation | 1.153 | **1.048** ᵃ | 1.245 |
| repaired @C1 | 0.505 | 0.785 | **0.807** ᵃ |
| repaired @C2 | −0.001 | −0.001 | **0.660** |
| repaired @C1P | −0.001 | −0.002 | **0.764** ᵃ |

ᵃ mean on the passing side of the threshold, but the seeds straddle it — by
§8's rule that decides nothing.

**Two findings carry.** First, [ADR-0010](docs/adr/0010-training-on-confident-errors-did-not-work.md)'s
verdict is withdrawn: training on confident errors *works*, moving C1P repair
from −0.002 to 0.764 ± 0.041 and engagement inside the region from 0.065 to
1.524. The 5k null was underpowering — a risk that was flagged in advance and
then reasoned past.

Second, and less obvious: **C1P teaches what C2 cannot.** `base 15k` trained on
C2 for 7,000 steps and still repairs it at −0.001; `ce 15k` saw the same C2 plus
C1P and repairs C2 at 0.660. Training on a corruption does not teach the model
to see it — training on the permuted one does, and it transfers. C1P preserves a
row's distribution exactly, so the only error is the assignment; C2 confounds
assignment with a change in shape.

The clean-fraction hypothesis is **refuted**: lowering it made every repair
metric worse and unstable (C2 0.436 ± 0.390, C1P 0.524 ± 0.456).

### G1 (P1): the recurrence is useful

**Three seeds per arm**, as §8 requires. Both arms trained identically — 5,000
steps, 400 meshes, lr 1e-3 — and each evaluated at the depth it was trained at,
on 8 held-out meshes from a C1 local-patch corruption.

| | parameters | weight L1 @T=8 | deformation @T=8 |
|---|---|---|---|
| **shared cell, k=4** | **33,665** | **0.0321 ± 0.0019** | **0.00038 ± 0.00003** |
| B3: 8 unshared layers, k=8 | 302,985 (9.0×) | 0.0395 ± 0.0035 | 0.00045 ± 0.00002 |

| G1 condition | measured | threshold | seeds straddle? |
|---|---|---|---|
| recurrence gain `L1(T=8)/L1(T=1)` | 0.274 ± 0.016 | ≤ 0.80 | no |
| versus B3 at T=8 | 0.814 ± 0.048 | ≤ 1.05 | no |
| smaller than B3 | 9.0× fewer | — | — |

No seed lands on the wrong side of either threshold, so the conclusion does not
rest on a mean concealing disagreement. The shared cell is **19% better than B3
at 9× fewer parameters**: iterating one learned rule beats stacking nine
independent copies of it.

Both deformation figures were previously reported as "0.0004" — the same
four-decimal rounding that hid the G2 drift, understating G1 in the process. At
five decimals the shared cell is also **16% better on the reference metric**,
which is the stronger of the two claims: it wins on what a rig is for, not only
on the control.

A detail that is not incidental: B3's curve is non-monotone away from T=8
(0.166, 0.221, 0.204, **0.038**, 0.132). Only its final layer is trained to emit
a valid answer, so intermediate depths are worse than the input. The shared cell
improves at every T because **every state is a solution, not just the last one** —
which is the difference between a dynamics and a pipeline.

### G2 (P2): there is no attractor — it is a monotone flow

Measured directly rather than inferred from a ratio (3 seeds × 3 meshes, 1,024
steps past T=32):

| | mean | range |
|---|---|---|
| cos(consecutive step directions) | **0.999 ± 0.001** | [0.998, 1.000] |
| PC1 share of trajectory variance | 0.962 ± 0.029 | [0.920, 0.996] |
| components for 95% variance | 1.44 | [1, 2] |
| error growth from a **ground-truth** start | **52.9× ± 28.5** | [33.8, 120.8] |

A smooth, near-one-dimensional **monotone flow**. No oscillation, no limit
cycle, no torus. And no basins: over 2,048 steps no two trajectories converge —
GT and C3 sit 0.87 apart at the start and 0.83 apart at the end.

**Handed the ground truth, the cell moves away from it by 53×**, reaching weight
L1 0.186 by step 2,079 — worse than the C1 corruption (0.160) it exists to
repair. The 1.097 ratio at T=32 previously reported here was the first 3% of
that curve.

This closes off the comfortable reading. The NCA literature finds oscillatory
and quasi-periodic attractors rather than fixed points
([arXiv 2604.12720](https://arxiv.org/abs/2604.12720)), which would have meant
G2 asks for something this architecture family cannot give. It is not what our
system does, so **G2's failure is real rather than a mis-specified gate**. See
[ADR-0015](docs/adr/0015-the-dynamics-is-a-flow-not-an-attractor.md).

### G2, as previously characterised (superseded)

Both metrics fail, on every seed:

| metric | T=8 | T=32 | ratio | needs |
|---|---|---|---|---|
| weight L1 | 0.03213 | 0.03932 | 1.223 ± 0.058 | ≤ 1.05 |
| deformation | 0.00038 | 0.00044 | 1.153 ± 0.043 | ≤ 1.05 |

A 14-point grid of T from 4 to 128 on the same checkpoints says what shape the
failure has:

| | |
|---|---|
| minimum | T = 8 |
| excess over the minimum | `(T−8)^**1.15**`, R² = 0.990 |
| per-step growth, T=12 → 128 | flat at ~0.00031 L1/step |
| per-step state change (median) | flat at ~0.00016 L1/step |

**Constant velocity over T=12–128.** Not diffusive accumulation (which would
decelerate, `~√T`), not a growing instability. Past T=128 the rate does bend
upward — 0.00032/step at T=32→64 against 0.00047/step at T=128→256 — so the
constant-rate description is scoped to the range measured, not a claim about the
limit.

**What it drifts toward: nothing in particular.** Out to T=256, the shape of the
weight distribution barely moves — mean max weight 0.832 → 0.826, entropy 0.380
→ 0.393, L1 to uniform 1.570 → 1.564. Over the same span L1 to ground truth goes
0.00015 → 0.102, a factor of 680. The field is not smoothing toward uniform and
not sharpening: mass **migrates between candidate bones while staying equally
peaked**. Vertices slowly change allegiance. Started from a C1 corruption
instead, the cell repairs to 0.035 by T=8 and has undone most of that by T=256
(0.136, against an initial 0.174).

The leading explanation is mundane and should be ruled out first: the cell is
trained with k=4 and a single overflow step, so **nothing has ever asked it to
hold still at T=128**. Constant drift off the training manifold needs no theory
beyond that. A constant per-step bias is what a stability term penalises, so
**extending `L_stab` is the indicated fix** — see
[ADR-0009](docs/adr/0009-the-drift-is-constant-velocity.md).

Two corrections came out of measuring this properly, both recorded in that ADR:

* **Deformation is not blind to the drift.** It had been reported flat at 0.0004
  from T=4 to T=32; that was rounding, at four decimals, on a curve that lives
  entirely in the fourth. It fails G2 too. The consoling H3 reading — that the
  state drifts along directions the reference metric cannot see — is withdrawn.
  The rig genuinely degrades, just more slowly than L1 suggests.
* **`stability` was measured between consecutive checkpoints**, over a
  deliberately non-uniform grid, so a constant rate read as divergence: the raw
  column climbed 0.00038 → 0.0057 while the true rate never moved. It is now
  `stability_per_step`, and renamed so old curves cannot be compared to new ones
  by accident.

### G3 (P3): repair is partial on C1 and absent on C2

The protocol is not the C1 evaluation above. There, the ground truth is
corrupted and the cell solves from a cold start. P3 is a different claim: a
state that has already **converged** is damaged, and the dynamics runs on from
it. The hidden state is zeroed inside the damaged region, because a cell
recovering from its own memory of the answer is not repairing anything. See
[ADR-0008](docs/adr/0008-repair-protocol-and-the-drift-control.md).

| | induced L1 | moved inside Ω @T=8 | repaired @T=8 | needs |
|---|---|---|---|---|
| C1 random patch | 1.606 | 1.115 ± 0.233 | 0.505 ± 0.169 | ≥ 0.80 |
| C2 hierarchy transfer | 1.695 | **0.007 ± 0.004** | **−0.001 ± 0.001** | ≥ 0.80 |

C1 repair is real but **saturates**: 0.505 at T=8, 0.510 at T=16, 0.507 at T=32.
Not slow — stuck. Half the induced error is permanent and four times the step
budget buys nothing.

C2 is the more interesting failure. At the *same* damage magnitude the state
barely moves: 0.007 against C1's 1.115. The cell does not fail to repair C2 — it
does not try. A 2×2 probe (3 seeds) says why:

| damage | entropy in patch | neighbour agreement | moved inside | repaired |
|---|---|---|---|---|
| C1 random patch | 1.891 | 12% | 1.115 | 0.505 |
| C2 hierarchy transfer | 0.164 | 48% | 0.007 | −0.001 |
| **peaked-random** — wrong, inconsistent, confident | 0.000 | 13% | **0.000** | 0.000 |
| **flat-hierarchy** — C2's bone error, spread out | 1.920 | 60% | **0.883** | 0.441 |

Neighbour agreement does not predict repair; it is mildly *anti*-correlated.
**Entropy predicts it completely.** The cell engages with a flat state and
ignores a peaked one — whether or not the peaked one is correct, and whether or
not its neighbours agree.

The mechanism is a learned shortcut. Every corruption the cell was trained to
fix arrives flat (C0 adds noise, C1 writes uniform-random rows) and every state
it was trained to leave alone is peaked (the CLEAN fraction, R3's do-no-harm
objective). Peakedness is a near-perfect proxy for "needs work" across the
curriculum, so gradient descent found it instead of learning to judge
correctness. **A confidently wrong state is invisible to the model.**

Coverage is not the explanation, though it looks like one: C2 enters the
curriculum only at step 8,000 and every run here is 5,000 steps, so these models
never saw it. But `flat-hierarchy` and `peaked-random` were equally unseen, and
one is repaired at 0.441 while the other does not move — what separates them is
entropy, not familiarity. The shortcut *generalises*.

**The obvious fix was tried and it failed.** Spec §3.2 defines C1 as "weights
randomised **or permuted**" and only the randomised half was implemented; the
permuted half (`C1P`) preserves a row's shape exactly and moves it to the wrong
bone — a confident error by construction. Trained at half of C1's share, three
seeds:

| | default | confident-error |
|---|---|---|
| repaired @C1P | −0.001 ± 0.002 | **0.004 ± 0.006** (needs 0.80) |
| weight L1 @T=8 | **0.0321 ± 0.0019** | 0.0403 ± 0.0026 |
| deformation @T=8 | **0.00038 ± 0.00003** | 0.00050 ± 0.00005 |
| drift T=32/T=8 | 1.223 ± 0.058 | **1.083 ± 0.056** |

Training directly on confident errors did not teach the model to see them, and
cost 26% on weight L1 and 32% on deformation. The one thing that improved was
not the target: drift fell to 1.083, the largest movement any intervention has
produced on G2 — though whether that is the corruption's doing or a side effect
of the accuracy regression is not established. See
[ADR-0010](docs/adr/0010-training-on-confident-errors-did-not-work.md) for the
three readings still open; the leading one is that clean states are ~45% of the
pool and peaked by construction, so the signal saying *leave peaked states
alone* outvotes the new one 3:1.

An earlier version of this section attributed C2 to a *locally self-consistent*
patch that local rules cannot see. That was wrong and the probe refutes it — a
C2 patch has 4.8 distinct target bones and 41% neighbour disagreement.

**The collateral half is not a do-no-harm failure.** Raw collateral is
0.768 ± 0.143 against a 0.10 threshold, but the same state run the same number
of further steps with *no damage* drifts 0.734 ± 0.154. Only **+0.034** is
attributable to the damage, which passes. Repair does not smear into the correct
region (R3 holds); the state simply drifts at the constant rate G2 measures.
Absolute magnitudes are small: fed the ground truth, the complement region goes
0.0001 → 0.0031 by T=8 → 0.0045 by T=16, monotone and away.

**Two gates fail for one cause.** G3's collateral is G2's constant-velocity
drift on a smaller baseline, which concentrates the next experiment rather than
dividing it.

### A4: the damped-delta update is what makes it a dynamics

The spec's vital ablation, 3 seeds per arm, identical training and **identical
parameter count** — A4 changes the update rule, not the architecture.

| | deltas (design) | absolute (A4) |
|---|---|---|
| weight L1 @T=8 | **0.0321 ± 0.0019** | 0.1667 ± 0.0282 |
| deformation @T=8 | **0.00038 ± 0.00003** | 0.00177 ± 0.00032 |
| per-step stability @T=8 | **0.000203 ± 0.000093** | 0.008560 ± 0.000780 |

Weight L1 by T tells the story better than any single number:

| | T=0 | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|---|---|---|
| deltas | 0.1603 | 0.1173 | 0.0728 | 0.0368 | **0.0321** | 0.0339 | 0.0393 |
| absolute | 0.1603 | **0.7161** | 0.5882 | 0.1280 | 0.1667 | 0.1995 | 0.2228 |

The absolute arm's first step makes the input **4.5× worse** and it never
recovers to its own T=4 value. That is B3's pathology exactly — only the
endpoint is a valid answer — and A4 shows the *update rule* creates it, not
depth or capacity. `z ← z + α·Δz` is what makes every intermediate state a
solution.

**A caution the ablation exposed:** the absolute arm "passes" G1's gain half
(`L1(T=8)/L1(T=1) = 0.246`, better than the delta arm's 0.274) purely because
its T=1 is catastrophic. A ratio against a terrible starting point flatters a
terrible model, which is why G1 also requires the B3 comparison. The gain
figure should never be quoted alone.

**And it corrects [ADR-0009](docs/adr/0009-the-drift-is-constant-velocity.md).**
The absolute arm has no increment to accumulate and drifts *more* (1.349 vs
1.223). So G2's drift is a property of the cell's mapping — repeated application
moves the state away from the truth — not of the integrator. `L_stab` may still
suppress it, but the mechanism ADR-0009 assumed is ruled out. See
[ADR-0011](docs/adr/0011-a4-damped-deltas-confirmed.md).

### A6: Euclidean candidates confirmed — but H1 is still untested

| | coverage (probe) | deformation @T=8 | weight L1 @T=8 |
|---|---|---|---|
| Euclidean | mean 0.9984, p5 1.0000 | **0.00031 ± 0.00001** | **0.0234 ± 0.0007** |
| diffusion | mean 0.9320, **p5 0.3106** | 0.00412 ± 0.00003 | 0.0487 ± 0.0011 |

Diffusion ranking is **13× worse on deformation**, and conservatively so — each
arm is scored against the truth projected onto *its own* candidate set, so the
diffusion arm is measured against an easier target. [ADR-0006](docs/adr/0006-euclidean-candidates-on-d1.md)
stands, now after training rather than on oracle coverage alone.

**But A6 does not test H1.** `PAIR_FEATURE_DIM` is 7 and its second component is
diffusion distance — *both* arms already receive geodesic information as a
feature. A6 varies only which metric ranks the top-8 candidates. H1 claims
injected geodesy lets anatomical separation emerge; geodesy was never withheld
from either arm, so the ablation cannot speak to it. The real H1 test — drop
diffusion distance from the pair features, keep Euclidean ranking — is one arm,
~3h, no cache rebuild, and has not been run. See
[ADR-0014](docs/adr/0014-a6-and-g1-at-the-spec-budget.md).

### B5: three fixed rules do not suffice

Reynolds' rules hand-coded and grid-searched over 1,715 configurations:

| corruption | vertices affected | B0 (input) | B5 (tuned) |
|---|---|---|---|
| C0 global noise | 100% | 0.5885 | **0.4300** (−27%) |
| C1 local patch | 11% | 0.1602 | 0.1602 (**0%**) |

Fixed rules denoise globally but **cannot localise**. Under C1 the search picks
the identity: no coefficient helps, because a rule that repairs the corrupted
11% damages the correct 89%. That is the do-no-harm problem (R3) exactly, and
the reason repair needs a learned gate. The trained cell takes the same input
from 0.1602 to 0.0304.

B1 (inverse-distance²) scores 1.2477 — far worse than the input it ignores. It
is a from-scratch baseline; the G4 comparison the spec defines starts from a C3
naive initialisation, which is V0.5 work.

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
uv run flock repair --run v0_YYYYMMDD_my-experiment  # gate G3
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

| Tier | Source | Meshes | Ground truth | Status |
|---|---|---|---|---|
| D0 | procedural capsule rigs | 200-500 generated | analytic | **not built** — `flock data gen` is a stub |
| D1 | Articulation-XL 2.0, canonical subset | 2,992 | artist, one 22-joint skeleton | in use |

**Every gate verdict here rests on D1 alone.** Spec §5 sequences D0 first so that
G0-G3 are decided where the ground truth is analytic and any gap is
unambiguously the model's; that has not happened. The C2 blindness
(ADR-0008), the drift (ADR-0009) and the gate results are all measured against
artist weights, which H3 itself calls non-unique.

D1 is one skeleton but not one body. Across a 428-mesh sample the median
character is twice as deep relative to its height as a human (0.49 vs ~0.25),
6.3% are wider than tall, 8.6% carry a non-vertical spine, and leg length ranges
0.55-1.42x of spine. These are creatures, robots and monsters wearing a Mixamo
rig - which makes A6 (diffusion vs Euclidean candidates) a sharper test of H1
than a corpus of human bipeds would be, since bulky bodies are where Euclidean
neighbourhoods should fail.

D1 is selected out of Articulation-XL 2.0 (MagicArticulate, CVPR 2025) by three
rules in [articulation_xl.py](src/flock/data/acl/articulation_xl.py): Mixamo
naming, a core that reduces exactly to the canonical 22 joints, and at most 20%
of weight mass on fingers. 22.5% of the corpus survives, on **one** hierarchy.
Meshes failing the rules are dropped rather than normalised — see
[ADR-0005](docs/adr/0005-d1-tier-from-articulation-xl.md). Rule 2 admits no
other topology: no quadrupeds, no wings, no tails, no extra limbs. Every
conclusion in this repo is therefore about a single 22-joint bipedal skeleton.

## Target

Apple M4, 24 GB unified memory, MLX/Metal. Roughly one hour of useful training
per experiment. Note that the spec sizes its throughput budget against an M4
**Pro**; see [ADR-0003](docs/adr/0003-fixed-shapes-and-npz-schema.md).
