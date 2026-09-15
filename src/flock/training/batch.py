"""Framework-neutral batch preparation (spec §2.1).

What remains here after the device-side work moved behind the port
(`backends/mlx/bank.py`) is the part that is genuinely backend-independent: the
logit initialisation every backend needs and every corruption level produces.

Keeping it in `training/` rather than duplicating it per adapter is the point —
`log(w + eps)` is not a compute-device concern.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

LOGIT_EPSILON = 1e-6
"""Floor inside `log(w + eps)`, so a zero weight maps to a finite logit."""


def logits_from_weights(weights: NDArray[np.float32]) -> NDArray[np.float32]:
    """`z = log(w + eps)`, the initialisation of spec §2.1."""
    logits: NDArray[np.float32] = np.log(
        np.maximum(weights, 0.0) + LOGIT_EPSILON
    ).astype(np.float32)
    return logits
