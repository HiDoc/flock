"""Corruption curriculum C0–C4 (spec §3.2).

Corruption is always applied *after* features and candidates are computed. A
corruption function therefore takes no mesh features and cannot influence them —
the structural half of the anti-leakage rule (spec R7).

Every level returns the affected region Ω alongside the corrupted field. That
mask is what makes P3 measurable: repair is scored inside Ω and collateral
damage on its complement, and without the mask the two are indistinguishable
from simply re-solving the whole mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from flock.domain.geometry.skeleton import ROOT_PARENT
from flock.domain.skinning.weights import WeightField

C0_SIGMA = 0.15
"""Standard deviation of the C0 perturbation, before renormalisation."""

PATCH_FRACTION = (0.05, 0.15)
"""Share of real vertices a C1/C2 patch covers (spec §3.2)."""

C4_SIGMA = 0.25
"""Relative noise on the near-uniform C4 field, before renormalisation."""


class CorruptionLevel(StrEnum):
    """Curriculum levels, in increasing order of difficulty (spec §3.2)."""

    C0_GAUSSIAN_LOGITS = "c0"
    """Light global Gaussian noise on GT logits. V0."""

    C1_LOCAL_PATCH = "c1"
    """BFS patch of 5-15% of vertices, weights randomised or permuted. V0."""

    C1P_PERMUTED_PATCH = "c1p"
    """The other half of spec §3.2's C1 — "weights randomised **or permuted**".
    Only the randomised half was implemented, and the two are not
    interchangeable: randomising flattens a row, permuting preserves its shape
    exactly and moves it to the wrong bone. Measurement showed the cell repairs
    the flat kind and is blind to the peaked kind, so the omitted half is the
    only one that trains it to judge correctness rather than confidence
    (ADR-0008). Kept a separate level rather than folded into C1 so every
    result measured against C1 stays comparable."""

    C2_HIERARCHY_TRANSFER = "c2"
    """Mass moved to a hierarchical neighbour bone over a patch. V0."""

    C3_NAIVE_GEOMETRIC = "c3"
    """Inverse-square-distance initialisation over candidates. V0.5."""

    C4_NEAR_UNIFORM = "c4"
    """Near-uniform weights over candidates plus noise. V0.5 — the closest
    regime to genuine self-organisation."""

    CLEAN = "clean"
    """No corruption. Always a fraction of every batch, so the model is trained
    to leave correct states alone (spec R3, the do-no-harm objective)."""


@dataclass(frozen=True)
class CorruptionContext:
    """The mesh facts corruption needs, and nothing else.

    Deliberately not the whole `PreprocessedMesh`: corruption must not be able
    to read features or ground truth beyond the field it is handed.
    """

    neighbours: NDArray[np.int32]
    neighbour_mask: NDArray[np.bool_]
    vertex_mask: NDArray[np.bool_]
    candidate_bones: NDArray[np.int32]
    candidate_mask: NDArray[np.bool_]
    candidate_distances: NDArray[np.float32]
    bone_parents: NDArray[np.int32]


def _renormalise(values: NDArray[np.float64]) -> NDArray[np.float32]:
    """Restore the partition of unity `WeightField` requires."""
    clipped = np.clip(values, 0.0, None)
    total = clipped.sum(axis=1, keepdims=True)
    np.divide(clipped, total, out=clipped, where=total > 0)
    return clipped.astype(np.float32)


def grow_patch(
    context: CorruptionContext,
    fraction: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """A connected patch of roughly `fraction` of the real vertices, by BFS.

    Connected rather than random: a scattered set of corrupted vertices is
    repaired by pure smoothing, which would let P3 pass without the dynamics
    doing anything interesting. A contiguous region has an interior that no
    neighbour can see the answer from.
    """
    live = np.flatnonzero(context.vertex_mask)
    if live.size == 0:
        return np.zeros(context.vertex_mask.shape[0], dtype=bool)

    target = max(1, int(fraction * live.size))
    patch = np.zeros(context.vertex_mask.shape[0], dtype=bool)
    frontier = [int(rng.choice(live))]
    patch[frontier[0]] = True
    count = 1

    while frontier and count < target:
        current = frontier.pop(0)
        for slot in range(context.neighbours.shape[1]):
            if not context.neighbour_mask[current, slot]:
                continue
            neighbour = int(context.neighbours[current, slot])
            if not patch[neighbour] and context.vertex_mask[neighbour]:
                patch[neighbour] = True
                frontier.append(neighbour)
                count += 1
                if count >= target:
                    break
    return patch


def _hierarchy_neighbours(bone: int, parents: NDArray[np.int32]) -> list[int]:
    """The parent and children of `bone` — its neighbours in the chain."""
    found = [int(c) for c in np.flatnonzero(parents == bone)]
    parent = int(parents[bone])
    if parent != ROOT_PARENT:
        found.append(parent)
    return found


def corrupt(
    weights: WeightField,
    level: CorruptionLevel,
    context: CorruptionContext,
    rng: np.random.Generator,
) -> tuple[WeightField, NDArray[np.bool_]]:
    """Apply one curriculum level to a weight field.

    Args:
        weights: The field to corrupt, typically ground truth or a converged state.
        level: Which curriculum level to apply.
        context: The mesh facts the corruption needs.
        rng: Seeded generator; corruption must be reproducible per run.

    Returns:
        The corrupted field and a `[V]` bool mask of the affected region Ω.

    C3 and C4 are *initialisations* rather than corruptions: they ignore
    `weights` entirely and build a field from geometry alone. Gate G4 starts
    from C3, which is the only place in this project where the model is asked
    to produce a solution rather than repair one.
    """
    values = weights.values.astype(np.float64)
    nothing = np.zeros(values.shape[0], dtype=bool)

    if level is CorruptionLevel.CLEAN:
        return weights, nothing

    if level is CorruptionLevel.C0_GAUSSIAN_LOGITS:
        noisy = values + rng.normal(0.0, C0_SIGMA, values.shape)
        return WeightField(_renormalise(noisy)), context.vertex_mask.copy()

    if level is CorruptionLevel.C1_LOCAL_PATCH:
        patch = grow_patch(context, rng.uniform(*PATCH_FRACTION), rng)
        corrupted = values.copy()
        rows = np.flatnonzero(patch)
        if rows.size:
            corrupted[rows] = rng.random((rows.size, values.shape[1]))
        return WeightField(_renormalise(corrupted)), patch

    if level is CorruptionLevel.C1P_PERMUTED_PATCH:
        patch = grow_patch(context, rng.uniform(*PATCH_FRACTION), rng)
        corrupted = values.copy()
        for vertex in np.flatnonzero(patch):
            # Permute among the *valid* slots only: mass landing on a padding
            # slot would be weight on a bone that is not a candidate at all.
            slots = np.flatnonzero(context.candidate_mask[vertex])
            if slots.size > 1:
                corrupted[vertex, slots] = corrupted[vertex, rng.permutation(slots)]
        return WeightField(_renormalise(corrupted)), patch

    if level is CorruptionLevel.C2_HIERARCHY_TRANSFER:
        patch = grow_patch(context, rng.uniform(*PATCH_FRACTION), rng)
        corrupted = values.copy()
        rows = np.flatnonzero(patch)
        for vertex in rows:
            slots = context.candidate_bones[vertex]
            dominant = int(np.argmax(corrupted[vertex]))
            # A hierarchical neighbour is the parent *or* a child: the root has
            # no parent, and hip-dominated vertices are a large share of any
            # character, so parent-only would leave them uncorrupted.
            neighbours = _hierarchy_neighbours(int(slots[dominant]), context.bone_parents)
            available = [
                int(found[0])
                for bone in neighbours
                if (found := np.flatnonzero(slots == bone)).size
            ]
            if not available:
                continue
            # The confusion an artist actually makes: mass on the neighbouring
            # bone in the chain, not on an unrelated one.
            target = int(rng.choice(available))
            moved = corrupted[vertex, dominant]
            corrupted[vertex, dominant] = 0.0
            corrupted[vertex, target] += moved
        return WeightField(_renormalise(corrupted)), patch

    if level is CorruptionLevel.C3_NAIVE_GEOMETRIC:
        # A from-scratch initialisation, not a corruption of anything: `weights`
        # is deliberately unread. That makes the anti-leakage rule structural —
        # C3 *cannot* carry ground truth into gate G4, because it never sees any
        # (spec R7).
        near = np.maximum(context.candidate_distances.astype(np.float64), 1e-6)
        naive = np.where(context.candidate_mask, 1.0 / (near**2), 0.0)
        return WeightField(_renormalise(naive)), context.vertex_mask.copy()

    if level is CorruptionLevel.C4_NEAR_UNIFORM:
        # The regime closest to genuine self-organisation: every candidate
        # equally plausible, so nothing but the dynamics breaks the symmetry.
        # `weights` is unread here too.
        flat = context.candidate_mask.astype(np.float64)
        jittered = flat * (1.0 + rng.normal(0.0, C4_SIGMA, flat.shape))
        return WeightField(_renormalise(jittered)), context.vertex_mask.copy()

    raise NotImplementedError(f"unhandled corruption level {level.value}")
