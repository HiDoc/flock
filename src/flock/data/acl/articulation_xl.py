"""D1 — Articulation-XL 2.0, restricted to its convention-homogeneous core.

Articulation-XL 2.0 (MagicArticulate, CVPR 2025) holds 48,637 rigged models
filtered from Objaverse-XL, across 443 category labels. Used whole, it is
exactly the hazard spec R6 names: heterogeneous artist conventions that blur the
training signal, which is why §5 sequences procedural data first and Mixamo —
"conventions homogènes" — second.

This adapter takes the second of those and finds it *inside* this corpus. Across
the test split, 34.7% of rigs are named `mixamorig_*`, and once finger chains are
folded away 80.7% of those share one identical 22-joint skeleton — same names,
same hierarchy, one core topology over all 559 meshes. Scaled to the full corpus
that is on the order of 11k meshes on a single skeleton, against the ~80 the spec
budgeted for D1.

So the convention homogeneity is not imposed by us; it is selected for, and
everything failing the test is dropped rather than normalised into agreement.

Three selection rules, in order:

1. every joint name carries the `mixamorig` prefix;
2. stripping finger chains yields exactly `CANONICAL_JOINTS`;
3. finger chains carry at most `MAX_FINGER_MASS` of the weight.

Rule 3 is what makes rule 2 honest. Folding fingers into the hand is not a
convenience — at V=1024 a decimated character retains almost no finger geometry,
so finger weights would survive as noise attached to vertices that no longer
resolve them. But some models in this corpus *are* hands and gloves, where
fingers carry up to 58% of the mass; folding those would destroy the signal
rather than clean it. The cap excludes them, keeping 85% of otherwise-canonical
meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flock.domain.geometry.skeleton import ROOT_PARENT, Skeleton

JOINT_PREFIX = "mixamorig"

FINGER_TOKENS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
"""Tokens marking a joint as part of a finger chain."""

CANONICAL_JOINTS: tuple[str, ...] = (
    "Hips",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)
"""The Mixamo core skeleton, in canonical index order.

Hips first, so that the root is always bone 0. Every selected mesh is reindexed
into this order, which is what makes bone indices mean the same thing across the
whole tier — the property that lets a batch mix meshes at all.
"""

MAX_FINGER_MASS = 0.20
"""Reject a mesh whose finger chains carry more than this share of the weight."""

NUM_CANONICAL_JOINTS = len(CANONICAL_JOINTS)


@dataclass(frozen=True)
class SelectionReport:
    """Why a corpus entry was kept or dropped, for the manifest."""

    uuid: str
    kept: bool
    reason: str
    num_vertices: int = 0
    finger_mass: float = 0.0


def _strip_prefix(name: str) -> str:
    """`mixamorig_LeftHand` -> `LeftHand`, tolerating `:` and `_` separators."""
    text = str(name)
    if not text.startswith(JOINT_PREFIX):
        return text
    return text[len(JOINT_PREFIX):].lstrip("_:")


def is_finger(name: str) -> bool:
    """True if the joint belongs to a finger chain."""
    return any(token in str(name) for token in FINGER_TOKENS)


def uses_mixamo_convention(joint_names: list[str]) -> bool:
    """True if every joint carries the Mixamo prefix."""
    return bool(joint_names) and all(str(n).startswith(JOINT_PREFIX) for n in joint_names)


def core_joints(joint_names: list[str]) -> tuple[str, ...]:
    """The joint names left once finger chains are removed, prefix stripped."""
    return tuple(_strip_prefix(n) for n in joint_names if not is_finger(n))


def is_canonical(joint_names: list[str]) -> bool:
    """True if this rig reduces exactly to the canonical skeleton."""
    return uses_mixamo_convention(joint_names) and core_joints(joint_names) == CANONICAL_JOINTS


def finger_mass(entry: dict[str, Any]) -> float:
    """Share of total weight mass carried by finger joints."""
    names = list(entry["joint_names"])
    fingers = np.array([is_finger(n) for n in names])
    values = np.asarray(entry["skinning_weights_value"], dtype=np.float64)
    columns = np.asarray(entry["skinning_weights_col"], dtype=np.int64)
    total = values.sum()
    if total <= 0:
        return 1.0
    return float(values[fingers[columns]].sum() / total)


def select(entry: dict[str, Any]) -> SelectionReport:
    """Apply the three selection rules to one corpus entry."""
    uuid = str(entry.get("uuid", "?"))
    names = list(entry["joint_names"])
    num_vertices = int(entry["skinning_weights_shape"][0])

    if not uses_mixamo_convention(names):
        return SelectionReport(uuid, False, "not-mixamo", num_vertices)
    if core_joints(names) != CANONICAL_JOINTS:
        return SelectionReport(uuid, False, "non-canonical-core", num_vertices)
    mass = finger_mass(entry)
    if mass > MAX_FINGER_MASS:
        return SelectionReport(uuid, False, "finger-dominant", num_vertices, mass)
    return SelectionReport(uuid, True, "canonical", num_vertices, mass)


def _parent_of(entry: dict[str, Any], num_joints: int) -> NDArray[np.int64]:
    """Parent joint index per joint, `ROOT_PARENT` at the root."""
    parents = np.full(num_joints, ROOT_PARENT, dtype=np.int64)
    for head, tail in np.asarray(entry["bones"], dtype=np.int64):
        parents[tail] = head
    return parents


def fold_to_canonical(entry: dict[str, Any]) -> NDArray[np.float32]:
    """Dense `[V, 22]` weights over the canonical skeleton.

    Finger weight is accumulated onto the nearest non-finger ancestor — the hand
    — and the result is reindexed into `CANONICAL_JOINTS` order, so bone k means
    the same joint on every mesh in the tier.

    Rows are renormalised at the end. The README asks for this explicitly, and
    `WeightField` rejects anything that skips it.
    """
    names = list(entry["joint_names"])
    num_vertices, num_joints = (int(x) for x in entry["skinning_weights_shape"])
    parents = _parent_of(entry, num_joints)

    canonical_index = {name: i for i, name in enumerate(CANONICAL_JOINTS)}
    target = np.full(num_joints, -1, dtype=np.int64)
    for j in range(num_joints):
        node, guard = j, 0
        while node != ROOT_PARENT and guard <= num_joints:
            slot = canonical_index.get(_strip_prefix(names[node]))
            if slot is not None and not is_finger(names[node]):
                target[j] = slot
                break
            node = int(parents[node])
            guard += 1

    weights = np.zeros((num_vertices, NUM_CANONICAL_JOINTS), dtype=np.float64)
    rows = np.asarray(entry["skinning_weights_row"], dtype=np.int64)
    columns = np.asarray(entry["skinning_weights_col"], dtype=np.int64)
    values = np.asarray(entry["skinning_weights_value"], dtype=np.float64)
    mapped = target[columns]
    keep = mapped >= 0
    np.add.at(weights, (rows[keep], mapped[keep]), values[keep])

    sums = weights.sum(axis=1, keepdims=True)
    np.divide(weights, sums, out=weights, where=sums > 0)
    return weights.astype(np.float32)


def to_skeleton(entry: dict[str, Any]) -> Skeleton:
    """Build the canonical `Skeleton`, joint-centric and uniformly indexed.

    Each bone is identified with its child joint — the standard convention, and
    the one that makes the corpus's per-joint weights line up with our per-bone
    weights without a translation step. Bone k runs from the position of joint
    k's parent to joint k's own position; the root bone (Hips) is zero-length at
    the hip position, so point-segment distance to it is distance to the joint,
    which is what a root influence means.
    """
    names = list(entry["joint_names"])
    positions = np.asarray(entry["joints"], dtype=np.float32)
    parents = _parent_of(entry, len(names))

    slot_of_joint = {}
    for j, name in enumerate(names):
        stripped = _strip_prefix(name)
        if not is_finger(name) and stripped in CANONICAL_JOINTS:
            slot_of_joint[stripped] = j

    heads = np.zeros((NUM_CANONICAL_JOINTS, 3), dtype=np.float32)
    tails = np.zeros((NUM_CANONICAL_JOINTS, 3), dtype=np.float32)
    canonical_parents = np.full(NUM_CANONICAL_JOINTS, ROOT_PARENT, dtype=np.int32)
    canonical_index = {name: i for i, name in enumerate(CANONICAL_JOINTS)}

    for slot, name in enumerate(CANONICAL_JOINTS):
        j = slot_of_joint[name]
        tails[slot] = positions[j]
        parent_joint = int(parents[j])
        if parent_joint == ROOT_PARENT:
            heads[slot] = positions[j]
            continue
        heads[slot] = positions[parent_joint]
        canonical_parents[slot] = canonical_index[_strip_prefix(names[parent_joint])]

    return Skeleton(heads, tails, canonical_parents, CANONICAL_JOINTS)


class ArticulationXLAdapter:
    """Reads Articulation-XL 2.0 shards, keeping only the canonical subset."""

    tier = "d1"

    def discover(self, root: Path) -> list[Path]:
        """List the corpus `.npz` shards under `root`."""
        return sorted(root.glob("articulation_xlv2*.npz"))

    def load(self, path: Path) -> Any:
        """Load one shard as its array of per-model dictionaries.

        Note the memory cost: `articulation_xlv2_train.npz` expands to 67 GB from
        a single pickled object array and cannot be opened on this machine. The
        other three shards expand to 2.5, 15.2 and 0.8 GB and load normally,
        which is ample — V0 needs hundreds of meshes, not tens of thousands.
        """
        return np.load(path, allow_pickle=True)["arr_0"]

    def to_skeleton(self, raw: Any) -> Skeleton:
        """Convert one entry to the canonical skeleton."""
        return to_skeleton(raw)
