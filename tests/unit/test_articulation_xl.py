"""Selection and folding for the Articulation-XL adapter.

Built on a synthetic rig so the suite stays runnable without the 37 GB corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from flock.data.acl.articulation_xl import (
    CANONICAL_JOINTS,
    MAX_FINGER_MASS,
    NUM_CANONICAL_JOINTS,
    finger_mass,
    fold_to_canonical,
    is_canonical,
    select,
    to_skeleton,
)
from flock.domain.geometry.skeleton import ROOT_PARENT

PARENT_OF = {
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2", "Head": "Neck",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2", "RightArm": "RightShoulder", "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}


def make_entry(with_fingers: bool = True, finger_weight: float = 0.1, num_vertices: int = 6):
    """A synthetic mixamorig rig: the canonical joints, optionally plus a thumb."""
    names = list(CANONICAL_JOINTS)
    parent = dict(PARENT_OF)
    if with_fingers:
        names += ["LeftHandThumb1", "LeftHandThumb2"]
        parent["LeftHandThumb1"] = "LeftHand"
        parent["LeftHandThumb2"] = "LeftHandThumb1"

    index = {n: i for i, n in enumerate(names)}
    bones = [[index[parent[n]], index[n]] for n in names if n in parent]
    joints = np.arange(len(names) * 3, dtype=np.float64).reshape(len(names), 3)

    rows, cols, vals = [], [], []
    for v in range(num_vertices):
        if with_fingers and finger_weight > 0:
            rows += [v, v]
            cols += [index["LeftHand"], index["LeftHandThumb2"]]
            vals += [1.0 - finger_weight, finger_weight]
        else:
            rows.append(v)
            cols.append(index["LeftHand"])
            vals.append(1.0)

    return {
        "uuid": "synthetic",
        "joint_names": [f"mixamorig_{n}" for n in names],
        "joints": joints,
        "bones": np.array(bones, dtype=np.int64),
        "skinning_weights_row": np.array(rows, dtype=np.int32),
        "skinning_weights_col": np.array(cols, dtype=np.int32),
        "skinning_weights_value": np.array(vals, dtype=np.float64),
        "skinning_weights_shape": (num_vertices, len(names)),
    }


class TestSelection:
    def test_canonical_rig_is_kept(self) -> None:
        assert select(make_entry()).kept

    def test_non_mixamo_rig_is_rejected(self) -> None:
        e = make_entry()
        e["joint_names"] = [n.replace("mixamorig_", "custom_") for n in e["joint_names"]]
        report = select(e)
        assert not report.kept and report.reason == "not-mixamo"

    def test_rig_missing_a_canonical_joint_is_rejected(self) -> None:
        e = make_entry(with_fingers=False)
        e["joint_names"] = e["joint_names"][:-1]
        e["skinning_weights_shape"] = (6, len(e["joint_names"]))
        report = select(e)
        assert not report.kept and report.reason == "non-canonical-core"

    def test_finger_dominant_rig_is_rejected(self) -> None:
        """A hand or glove model: folding would destroy the signal, not clean it."""
        report = select(make_entry(finger_weight=0.6))
        assert not report.kept and report.reason == "finger-dominant"
        assert report.finger_mass > MAX_FINGER_MASS

    def test_finger_mass_is_measured(self) -> None:
        assert finger_mass(make_entry(finger_weight=0.25)) == pytest.approx(0.25)

    def test_is_canonical_ignores_finger_depth(self) -> None:
        assert is_canonical(make_entry(with_fingers=True)["joint_names"])
        assert is_canonical(make_entry(with_fingers=False)["joint_names"])


class TestFold:
    def test_finger_weight_lands_on_the_hand(self) -> None:
        w = fold_to_canonical(make_entry(finger_weight=0.1))
        hand = CANONICAL_JOINTS.index("LeftHand")
        assert w.shape == (6, NUM_CANONICAL_JOINTS)
        assert w[:, hand] == pytest.approx(1.0)

    def test_rows_are_renormalised(self) -> None:
        """The corpus README asks for this; WeightField rejects anything that skips it."""
        e = make_entry(finger_weight=0.1)
        e["skinning_weights_value"] = e["skinning_weights_value"] * 0.5
        assert fold_to_canonical(e).sum(axis=1) == pytest.approx(1.0)

    def test_no_negative_weight_is_produced(self) -> None:
        assert (fold_to_canonical(make_entry()) >= 0).all()


class TestSkeleton:
    def test_skeleton_is_canonical_and_uniform(self) -> None:
        a, b = to_skeleton(make_entry(True)), to_skeleton(make_entry(False))
        assert a.num_bones == NUM_CANONICAL_JOINTS == 22
        assert a.names == CANONICAL_JOINTS
        assert a.parents.tolist() == b.parents.tolist(), "finger depth must not alter the core"

    def test_hips_is_the_single_root(self) -> None:
        sk = to_skeleton(make_entry())
        assert sk.names[sk.root] == "Hips"
        assert (sk.parents == ROOT_PARENT).sum() == 1

    def test_hierarchy_depths_are_anatomical(self) -> None:
        d = to_skeleton(make_entry()).depths()
        names = CANONICAL_JOINTS
        assert d[names.index("Hips")] == 0
        assert d[names.index("Head")] > d[names.index("Neck")]
        assert d[names.index("LeftHand")] > d[names.index("LeftShoulder")]
