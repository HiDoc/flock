"""Training loop (spec §3.1, §3.4).

Drives the six-step pool cycle: sample 32 slots, reinitialise a fraction with
fresh corruption and a fraction with clean GT, recorrupt a fraction of converged
states, roll out k=4 steps with gradients, update, write the states back
detached.

Budget (spec §3.4): 10-15k optimiser steps in 40-50 minutes. Note that the spec
sizes this against an M4 Pro; this machine is an M4 base, so the step-rate
target is re-measured at G0.5 rather than inherited (ADR-0003).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flock.domain.dynamics.state import CellConfig
from flock.training.losses import LossWeights


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters (spec §3.4)."""

    cell: CellConfig = field(default_factory=CellConfig)
    losses: LossWeights = field(default_factory=LossWeights)

    batch_size: int = 16
    """16, not the spec's 32: mesh-steps/s is flat across batch size on this
    machine, so 16 is the same compute for roughly twice the optimiser steps
    (ADR-0007)."""

    bptt_steps: int = 4
    """k: gradients flow through 4 steps; the pool supplies the long horizon."""

    pool_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    max_steps: int = 15_000
    seed: int = 0
    backend: str = "mlx"

    precision: str = "fp32"
    """fp32 first. bf16 only if profiling shows an actual throughput wall —
    not preemptively (spec §3.4)."""

    poses_per_step: int = 2
    """Poses drawn per step for L_deform (spec §3.3: 2-4 from a bank of 16-32).
    Posing the whole bank costs several times the rollout for no extra signal."""

    log_every: int = 50
    """Metrics are pulled off the GPU in batches; reading them every step would
    serialise the pipeline (spec §3.5)."""

    checkpoint_every: int = 500
    """A long run that dies partway should still leave something evaluable."""


def train(
    config: TrainConfig,
    run_name: str,
    cache: Path = Path("data/cache/d1"),
    runs_root: Path = Path("runs"),
    num_meshes: int = 400,
) -> None:
    """Run a full training job and write its artifacts to the run directory.

    Drives the six-step cycle of spec §3.1 — sample, reset a fraction, recorrupt
    a fraction, roll out k steps with gradients, update, write back detached.
    """
    import time

    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np

    from flock.backends.mlx.cell import init_params
    from flock.backends.mlx.train_step import make_pool_train_step
    from flock.data.store import DatasetRepository
    from flock.domain.dynamics.pool import StatePool
    from flock.domain.geometry.mesh import NUM_CANDIDATES, NUM_VERTICES
    from flock.domain.skinning.corruption import CorruptionLevel, corrupt
    from flock.experiment.repository import RunRepository
    from flock.training.batch import MeshBank, logits_from_weights
    from flock.training.curriculum import default_curriculum, sample_levels

    repo = DatasetRepository(cache)
    ids = repo.list_ids()[:num_meshes]
    samples = [repo.load(i)[0] for i in ids]
    bank = MeshBank(samples)
    contexts = [s.corruption_context() for s in samples]
    truth = [s.ground_truth for s in samples]

    runs = RunRepository(runs_root)
    runs.create(run_name, {"config": repr(config), "meshes": len(samples)})

    rng = np.random.default_rng(config.seed)
    pool = StatePool(
        len(samples), NUM_VERTICES, config.cell.hidden_dim, NUM_CANDIDATES,
        size=config.pool_size, seed=config.seed,
    )
    stages = default_curriculum()

    params = init_params(config.cell, config.seed)
    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    step_fn = make_pool_train_step(config.cell, config.bptt_steps, optimizer)

    started = time.time()
    for step in range(1, config.max_steps + 1):
        plan = pool.plan_batch(config.batch_size)
        meshes = pool.mesh_index[plan.slots]

        # Rewrite the fractions this step calls for, before reading the pool.
        rewrite = np.flatnonzero(plan.reset | plan.clean | plan.recorrupt)
        if rewrite.size:
            levels = sample_levels(stages, step, rewrite.size, rng)
            fresh = np.zeros((rewrite.size, NUM_VERTICES, NUM_CANDIDATES), dtype=np.float32)
            applied: list[CorruptionLevel] = []
            for row, position in enumerate(rewrite):
                mesh = int(meshes[position])
                level = CorruptionLevel.CLEAN if plan.clean[position] else levels[row]
                corrupted, _ = corrupt(truth[mesh], level, contexts[mesh], rng)
                fresh[row] = corrupted.values
                applied.append(level)
            pool.reset_slots(plan.slots[rewrite], logits_from_weights(fresh), applied)

        hidden, logits = pool.read(plan.slots)
        poses = rng.choice(
            samples[0].train_poses.num_poses, size=config.poses_per_step, replace=False
        )
        static, extras = bank.gather(meshes, poses)
        payload = {
            "hidden": mx.array(hidden),
            "logits": mx.array(logits),
            "w_weight": mx.array(config.losses.weight_l1),
            "w_deform": mx.array(config.losses.deformation_at(step)),
            "w_stability": mx.array(config.losses.stability),
            **extras,
        }
        params, total, new_hidden, new_logits, parts = step_fn(params, payload, static)

        if step % config.log_every == 0 or step == config.max_steps:
            mx.eval(params, total, new_hidden, new_logits, *parts)
            runs.record_metrics(run_name, {
                "step": step,
                "loss": float(total),
                "l_weight": float(parts[0]),
                "l_deform": float(parts[1]),
                "l_stability": float(parts[2]),
                "seconds": time.time() - started,
                **pool.age_histogram(),
                **pool.level_counts(),
            })
        if step % config.checkpoint_every == 0:
            # A run that dies at step 2000 of 6000 should still be evaluable.
            # Metrics already survive that way; the checkpoint has to as well.
            mx.eval(params)
            runs.save_checkpoint(run_name, params)
        # States go back detached: gradients span k steps, the dynamics spans the pool.
        pool.write(plan.slots, np.array(new_hidden), np.array(new_logits), config.bptt_steps)

    mx.eval(params)
    runs.save_checkpoint(run_name, params)


def overfit_single_mesh(
    config: TrainConfig,
    mesh_id: int,
    cache: Path = Path("data/cache/d1"),
    time_budget_seconds: float = 600.0,
) -> dict[str, float]:
    """Gate G0.5: drive one mesh to near-zero L1 (spec §6).

    A sanity check on the loop and the gradients, not a result. If this does not
    converge in under ten minutes, something is broken and no later number can
    be trusted.

    Returns:
        Metrics: final L1, the starting L1 it had to beat, steps taken, elapsed
        seconds and measured throughput.
    """
    import time

    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np

    from flock.backends.mlx.cell import init_params
    from flock.backends.mlx.train_step import make_train_step
    from flock.data.store import DatasetRepository
    from flock.domain.skinning.corruption import CorruptionLevel, corrupt
    from flock.training.batch import build_batch

    repo = DatasetRepository(cache)
    sample, _ = repo.load(mesh_id)
    rng = np.random.default_rng(config.seed)

    # One mesh, several independent corruptions of it: enough variety that
    # success means the dynamics learned something, not that it memorised a
    # single input vector.
    copies = 8
    corrupted = np.stack(
        [
            corrupt(
                sample.ground_truth,
                CorruptionLevel.C0_GAUSSIAN_LOGITS,
                sample.corruption_context(),
                rng,
            )[0].values
            for _ in range(copies)
        ]
    )
    batch = build_batch([sample] * copies, corrupted)
    tree = batch.static.as_tree()
    payload = {"initial_logits": batch.initial_logits, "target": batch.target}

    live = sample.mesh.vertex_mask
    start_l1 = float(
        np.abs(corrupted - sample.ground_truth.values[None])[:, live].sum(axis=-1).mean()
    )

    params = init_params(config.cell, config.seed)
    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    step = make_train_step(config.cell, config.bptt_steps, optimizer)

    started, loss = time.time(), mx.array(0.0)
    taken = 0
    for taken in range(1, config.max_steps + 1):
        params, loss = step(params, payload, tree)
        # Metrics are pulled off device in batches; reading every step would
        # serialise the pipeline (spec §3.5).
        if taken % config.log_every == 0:
            mx.eval(params, loss)
            if time.time() - started > time_budget_seconds:
                break
    mx.eval(params, loss)

    elapsed = time.time() - started
    return {
        "final_l1": float(loss),
        "start_l1": start_l1,
        "steps": float(taken),
        "seconds": elapsed,
        "steps_per_second": taken / max(elapsed, 1e-9),
    }
