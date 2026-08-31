"""Loss kernels (spec §3.3).

The *policy* — which losses, at what relative weight, with what warmup — lives
in `flock.training.losses`, because that is a statement about the experiment.
What lives here is the arithmetic, written in raw `mx` so it compiles into the
same graph as the rollout (ADR-0002).

| Loss | Weight | Role |
|---|---|---|
| `L_w` | 1.0 | L1 to GT, supervised only at the *last* rollout step |
| `L_deform` | 1.0 after warmup | LBS error under sampled poses — the loss that counts |
| `L_stab` | 0.05 | forces the change to zero at the fixed point |

There is no Laplacian regulariser. Message passing is supposed to produce
smoothness on its own; if it does not, that is a finding about the architecture,
and smoothing it away would hide exactly the thing V0 exists to measure.
"""

from __future__ import annotations

import mlx.core as mx

from flock.backends.mlx.lbs import lbs


def weight_loss(predicted_logits: mx.array, target_weights: mx.array, mask: mx.array) -> mx.array:
    """L1 between predicted and target weights at the final rollout step.

    Padded rows are excluded rather than averaged in; counting them would scale
    the loss by the padding ratio and make meshes of different sizes
    incomparable.
    """
    per_vertex = mx.sum(mx.abs(predicted_logits - target_weights), axis=-1)
    return mx.sum(per_vertex * mask) / mx.maximum(mx.sum(mask), 1.0)


def deformation_loss(
    positions: mx.array,
    predicted_weights: mx.array,
    target_weights: mx.array,
    bones: mx.array,
    transforms: mx.array,
) -> mx.array:
    """LBS vertex error over sampled poses.

    2-4 poses drawn per step from the precomputed bank; one einsum, effectively
    free next to the rollout.

    This is the loss that counts (H3). Weight L1 punishes correct solutions that
    happen to disagree with an artist's convention; this measures the only thing
    a rig is judged on.
    """
    moved = lbs(positions, predicted_weights, bones, transforms)
    reference = lbs(positions, target_weights, bones, transforms)
    return mx.mean(mx.sqrt(mx.sum(mx.square(moved - reference), axis=-1) + 1e-12))


def stability_loss(weights_t: mx.array, weights_next: mx.array, mask: mx.array) -> mx.array:
    """L1 change over 1-2 overflow steps past the supervised one.

    Runs *beyond* the supervised horizon on purpose: that is where the fixed
    point has to hold, and where nothing else is pushing it to. Without this the
    model has no reason to be quiet after the step it is graded on — which is
    the quiescence Reynolds' rules never needed (spec §0.2).
    """
    per_vertex = mx.sum(mx.abs(weights_next - weights_t), axis=-1)
    return mx.sum(per_vertex * mask) / mx.maximum(mx.sum(mask), 1.0)
