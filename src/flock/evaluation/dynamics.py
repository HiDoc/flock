"""Character of the learned dynamics (cf. Kvalsund & Stovold, arXiv 2604.12720).

Gate G2 asks *whether* the dynamics settles. These measures ask *what kind* of
dynamics it is, which turns out to be the more useful question: an oscillation
that never quiesces and a monotone flow that never quiesces fail G2 identically
and need opposite fixes.

The NCA literature reports oscillatory and quasi-periodic attractors, measured
with Lyapunov spectra and Fourier analysis. A largest-Lyapunov estimate was
tried here and **abandoned**: the state crosses the backend port as NumPy, so it
is quantised to float32 every step, and a two-trajectory estimate never reaches
a linear regime — the exponent scaled with the perturbation size (0.34, 0.16,
0.07 for eps 3e-4, 1e-3, 3e-3) rather than converging. The measures below use
large-amplitude, quantisation-robust quantities instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FlowCharacter:
    """What shape the trajectory traces through state space.

    Attributes:
        cos_consecutive: Mean cosine between successive step directions. Near 1
            is smooth monotone motion; near 0 is jitter; negative is
            oscillation, where the state reverses each step.
        cos_endpoints: Cosine between the first and last step direction. Near 1
            means a straight line, lower means the path curves — but read it
            with `cos_consecutive`, since a slow smooth curve and a cycle both
            lower it.
        pc1_share: Fraction of trajectory variance on the leading principal
            component. High means the motion is effectively one-dimensional.
        dimension_95: Principal components needed for 95% of the variance. A
            limit cycle needs 2, a k-torus needs k.
    """

    cos_consecutive: float
    cos_endpoints: float
    pc1_share: float
    dimension_95: int


def flow_character(trajectory: NDArray[np.float64]) -> FlowCharacter:
    """Characterise a state trajectory.

    Args:
        trajectory: `[T, D]` states in visit order — for FlockSkin, the weight
            field of the real vertices flattened, one row per iteration.

    Returns:
        The measures above.

    Raises:
        ValueError: If fewer than three states are supplied, which is too few
            for a step-to-step comparison.
    """
    if trajectory.shape[0] < 3:
        raise ValueError(f"need at least 3 states, got {trajectory.shape[0]}")

    steps = np.diff(trajectory, axis=0)
    norms = np.linalg.norm(steps, axis=1, keepdims=True)
    units = steps / np.maximum(norms, 1e-30)
    consecutive = float(np.sum(units[:-1] * units[1:], axis=1).mean())
    endpoints = float(units[0] @ units[-1])

    singular = np.linalg.svd(trajectory - trajectory.mean(axis=0), compute_uv=False)
    energy = singular**2
    total = energy.sum()
    if total <= 0:
        return FlowCharacter(consecutive, endpoints, 1.0, 1)
    cumulative = np.cumsum(energy) / total
    return FlowCharacter(
        cos_consecutive=consecutive,
        cos_endpoints=endpoints,
        pc1_share=float(cumulative[0]),
        dimension_95=int(np.searchsorted(cumulative, 0.95) + 1),
    )


def trajectories_converge(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    candidates: int = 8,
    ratio: float = 0.5,
) -> tuple[bool, float, float]:
    """Whether two trajectories fall into the same attractor.

    The direct test for "is there one attractor or several", and robust to the
    float32 round-trip because it works at the amplitude of genuinely different
    initial conditions rather than an infinitesimal perturbation.

    Args:
        first: `[T, D]` trajectory.
        second: `[T, D]` trajectory over the same iterations.
        candidates: Weights per vertex, for reshaping `D` into per-vertex rows.
        ratio: Fraction of the initial separation that counts as convergence.

    Returns:
        Whether they converged, plus the separation at the start and the end.
    """
    def gap(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(np.abs(a - b).reshape(-1, candidates).sum(axis=-1).mean())

    start, end = gap(first[0], second[0]), gap(first[-1], second[-1])
    return end < start * ratio, start, end
