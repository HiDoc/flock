# ADR-0005 — D1 comes from Articulation-XL 2.0, filtered to one convention

**Status:** accepted · **Date:** 2026-08-31

## Context

[The spec §5](../spec.md) defines D1 as "50–100 Mixamo characters, decimated to
~1k vertices, homogeneous skeleton", justified by risk **R6**: heterogeneous
artist conventions blur the training signal, so V0 should start on procedural
data with analytic ground truth and move to Mixamo, whose conventions are
consistent.

Articulation-XL 2.0 (MagicArticulate, CVPR 2025) offers 48,637 rigged models
filtered from Objaverse-XL, already preprocessed into NPZ with vertices, faces,
normals, joints, bones and sparse skinning weights. Used whole it is precisely
what R6 warns against: 443 category labels, from characters to weapons to
anatomy, each with whatever convention its author used.

The question was whether it can serve V0 without reintroducing that hazard.

## What the corpus actually contains

Measured over the 1,997-model test split:

| | |
|---|---|
| rigs named `mixamorig_*` | 693 / 1997 (34.7%) |
| of those, reducing to one core skeleton once finger chains are stripped | 559 / 693 (80.7%) |
| distinct core topologies among those 559, in name space | **1** |
| max influences per vertex, after folding fingers | **4** (against a B=8 budget) |
| weight rows failing the sum-to-1 check | 0 / 1997 |

The corpus contains a large, exactly homogeneous Mixamo population — same joint
names, same hierarchy, one topology — and it is identifiable from the data
rather than asserted.

## Decision

**D1 is the canonical subset of Articulation-XL 2.0**, selected by three rules in
`data/acl/articulation_xl.py`:

1. every joint name carries the `mixamorig` prefix;
2. stripping finger chains yields exactly the 22-joint `CANONICAL_JOINTS`;
3. finger chains carry at most 20% of the weight mass.

Selected meshes are reindexed into canonical order, so bone *k* denotes the same
joint on every mesh in the tier — the property that lets a batch mix meshes at
all. Everything failing the rules is **dropped, not normalised**: the homogeneity
is selected for, never imposed, because imposing it would manufacture exactly the
convention agreement R6 says we must not assume.

Two consequences of the corpus's shape:

* **Bones are modelled joint-centrically**, each bone identified with its child
  joint. This is the standard convention and it dissolves a mismatch — the corpus
  stores weights per joint (N×51) while `bones` is a list of joint-index pairs —
  without a translation step. It also makes Hips the single root that `Skeleton`
  requires; a bone-centric reading would give three roots, since Hips begins the
  spine and both legs.
* **Finger chains are folded into the hand.** At V=1024 a decimated character
  retains almost no finger geometry, so finger weights would persist as noise on
  vertices that can no longer resolve them. Rule 3 is what keeps this honest:
  some models in this corpus *are* hands and gloves, where fingers carry up to
  58% of the mass, and folding those would destroy the signal rather than clean
  it. The cap excludes them and keeps 85% of otherwise-canonical meshes.

`data/acl/mixamo.py` is removed. It was an unimplemented stub for the same tier,
and this adapter supersedes it — the Mixamo-convention data the spec wanted is
inside this corpus.

## Consequences

The tier is far larger than the spec budgeted — hundreds of meshes per shard
against the planned ~80 — while being *more* homogeneous than raw Mixamo
downloads, since the selection is verified per mesh rather than assumed from the
source. The `category_label` column of the metadata CSV still supplies the
held-out category the §5 split calls for.

**The ordering in §5 is unchanged.** D0 procedural still comes first: its ground
truth is analytic, so gates G0–G3 are settled where any gap is the model's rather
than the data's. D1 is what follows, and this decision only changes where D1
comes from.

**One shard is unreadable here.** `articulation_xlv2_train.npz` expands from 31 GB
to **67 GB** as a single pickled object array — it cannot be opened on a 26 GB
machine, and no amount of laziness helps, since numpy unpickles an object array
whole. The other three shards expand to 2.5, 15.2 and 0.8 GB and load normally.
Reaching the remaining ~46k models would need either a streaming unpickler or a
larger machine; neither is needed for V0, which is sized in hundreds of meshes.
