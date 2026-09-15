"""Command-line entry point.

The command surface follows the milestone order of the plan, so that
`flock --help` reads as the project's own critical path:

    flock data gen          generate D0
    flock data preprocess   build the .npz cache
    flock oracle coverage   gate G0 — run before training anything
    flock train             M2/M3
    flock eval              metrics by iteration over the probe set
    flock ablate            A1-A9
    flock gates             print the gate status table
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import numpy as np
import typer

from flock.evaluation.gates import GATES

app = typer.Typer(help="FlockRig / FlockSkin research CLI", no_args_is_help=True)
data_app = typer.Typer(help="Dataset generation and preprocessing", no_args_is_help=True)
oracle_app = typer.Typer(help="Pre-training diagnostics", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(oracle_app, name="oracle")

DEFAULT_CACHE = Path("data/cache")
DEFAULT_RUNS = Path("runs")


def _pending(milestone: str, what: str) -> NoReturn:
    """Report an unimplemented command as a clear message, not a traceback.

    Every command in the surface is wired from M0 so that `flock --help` reads
    as the project's critical path. Reaching one early should say which
    milestone fills it in, and exit.
    """
    typer.secho(f"{what}: not implemented yet — lands at {milestone}.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


@data_app.command("gen")
def data_gen(
    count: int = typer.Option(300, help="Number of procedural rigs to generate."),
    seed: int = typer.Option(0),
    out: Path = typer.Option(DEFAULT_CACHE),
) -> None:
    """Generate the D0 procedural dataset."""
    _pending("M1", "flock data gen")


@data_app.command("preprocess")
def data_preprocess(
    source: str = typer.Option("d1", help="Dataset tier: d0, d1 or d2."),
    root: Path = typer.Option(..., help="Directory holding the raw rigs."),
    out: Path = typer.Option(DEFAULT_CACHE),
    metric: str = typer.Option("euclidean", help="Candidate ranking: euclidean or diffusion."),
    limit: int = typer.Option(0, help="Stop after this many meshes (0 = all)."),
) -> None:
    """Preprocess raw rigs into the fixed-shape .npz cache."""
    from flock.data.acl.articulation_xl import select
    from flock.data.preprocess.pipeline import PreprocessConfig, preprocess_entry
    from flock.data.store import DatasetRepository

    if source != "d1":
        _pending("M1", f"flock data preprocess --source {source}")
    repo = DatasetRepository(out)
    config = PreprocessConfig(candidate_metric=metric)
    mesh_id, failures = 0, 0
    for shard in sorted(root.glob("articulation_xlv2*.npz")):
        if "train.npz" in shard.name and shard.stat().st_size > 20_000_000_000:
            typer.secho(
                f"skipping {shard.name}: expands to 67GB, unloadable here (ADR-0005)",
                fg=typer.colors.YELLOW,
            )
            continue
        entries = np.load(shard, allow_pickle=True)["arr_0"]
        for entry in entries:
            if not select(entry).kept:
                continue
            if limit and mesh_id >= limit:
                break
            try:
                result = preprocess_entry(entry, mesh_id, config, seed=mesh_id)
            except Exception as error:
                # One malformed mesh must not abort a 3,000-mesh tier.
                failures += 1
                typer.secho(f"  mesh {mesh_id}: {error}", fg=typer.colors.RED)
                continue
            repo.save(result.sample, result.dense_weights)
            mesh_id += 1
        typer.echo(f"{shard.name}: cache now {mesh_id} meshes")
        if limit and mesh_id >= limit:
            break
    typer.secho(f"wrote {mesh_id} meshes to {out} ({failures} failed)", fg=typer.colors.GREEN)


@oracle_app.command("coverage")
def oracle_coverage(
    cache: Path = typer.Option(DEFAULT_CACHE),
    source: str = typer.Option("d0"),
) -> None:
    """Gate G0: does the candidate set contain the ground-truth weight mass?

    Run this before any training. A coverage shortfall is a ceiling the model
    cannot cross, and every later number would inherit it.
    """
    from flock.data.store import DatasetRepository
    from flock.domain.skinning.metrics import coverage as coverage_metric

    repo = DatasetRepository(cache)
    ids = repo.list_ids()
    if not ids:
        typer.secho(f"no cached meshes in {cache}; run `flock data preprocess` first", fg="red")
        raise typer.Exit(code=1)

    means, fifths = [], []
    for mesh_id in ids:
        sample, dense = repo.load(mesh_id)
        values = coverage_metric(dense, sample.candidates)[sample.mesh.vertex_mask]
        means.append(float(values.mean()))
        fifths.append(float(np.percentile(values, 5)))

    mean, p5 = float(np.mean(means)), float(np.mean(fifths))
    per_mesh = sum(1 for m, f in zip(means, fifths, strict=True) if m >= 0.98 and f >= 0.90)
    passed = mean >= 0.98 and p5 >= 0.90

    typer.echo(f"meshes            : {len(ids)}  (source {source})")
    typer.echo(f"coverage mean     : {mean:.4f}   (G0 needs >= 0.98)")
    typer.echo(f"coverage p5       : {p5:.4f}   (G0 needs >= 0.90)")
    typer.echo(f"per-mesh passing  : {per_mesh}/{len(ids)}")
    typer.secho(
        f"GATE G0: {'PASS' if passed else 'FAIL'}",
        fg=typer.colors.GREEN if passed else typer.colors.RED,
        bold=True,
    )
    if not passed:
        typer.echo("Fix candidate selection before training anything (spec §4.4, R1).")
        raise typer.Exit(code=1)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/base.yaml")),
    change: str = typer.Option(..., help="The single thing this run changes, kebab-case."),
    seed: int = typer.Option(0),
    overfit: int | None = typer.Option(None, help="Gate G0.5: overfit this mesh id alone."),
    cache: Path = typer.Option(Path("data/cache/d1")),
    lr: float = typer.Option(3e-3, help="Higher than the 3e-4 default; G0.5 is a sprint."),
    steps: int = typer.Option(4000),
    meshes: int = typer.Option(400, help="Training meshes to hold resident."),
    unshared: bool = typer.Option(False, help="A2/B3: give every step its own parameters."),
    absolute: bool = typer.Option(False, help="A4: predict logits outright, not damped deltas."),
    bptt: int = typer.Option(4, help="k: rollout length gradients flow through."),
    confident_error: bool = typer.Option(
        False, help="Train on C1P, the peaked corruption, for half of C1's share."
    ),
    clean_share: float = typer.Option(
        -1.0, help="Override the curriculum's clean share; negative keeps the default."
    ),
) -> None:
    """Train the cell, or run the single-mesh sanity gate."""
    from flock.domain.dynamics.state import CellConfig
    from flock.training.loop import TrainConfig, overfit_single_mesh

    if overfit is None:
        from flock.experiment.repository import make_run_name
        from flock.training.loop import train as run_training

        name = make_run_name("v0", change)
        settings = TrainConfig(
            cell=CellConfig(shared_weights=not unshared, absolute_logits=absolute),
            seed=seed, learning_rate=lr, max_steps=steps, bptt_steps=bptt,
            confident_error=confident_error,
            clean_share=clean_share if clean_share >= 0 else None,
        )
        typer.echo(f"run {name}: {steps} steps, batch {settings.batch_size}, {meshes} meshes")
        run_training(settings, name, cache=cache, num_meshes=meshes)
        typer.secho(f"wrote runs/{name}", fg=typer.colors.GREEN)
        return

    settings = TrainConfig(
        cell=CellConfig(),
        seed=seed,
        learning_rate=lr,
        max_steps=steps,
        batch_size=8,
    )
    typer.echo(f"G0.5: overfitting mesh {overfit} (k={settings.bptt_steps}, lr={lr})")
    result = overfit_single_mesh(settings, overfit, cache=cache)

    ratio = result["final_l1"] / max(result["start_l1"], 1e-9)
    typer.echo(f"start L1          : {result['start_l1']:.4f}")
    typer.echo(f"final L1          : {result['final_l1']:.4f}  ({ratio:.1%} of start)")
    typer.echo(f"steps             : {int(result['steps'])} in {result['seconds']:.1f}s")
    typer.echo(f"throughput        : {result['steps_per_second']:.2f} steps/s (batch 8, k=4)")
    passed = result["final_l1"] < 0.05 and result["seconds"] < 600
    typer.secho(
        f"GATE G0.5: {'PASS' if passed else 'FAIL'}",
        fg=typer.colors.GREEN if passed else typer.colors.RED,
        bold=True,
    )
    if not passed:
        typer.echo("Loop or gradient bug. Stop and fix before anything else (spec §6).")
        raise typer.Exit(code=1)


@app.command("eval")
def evaluate(
    run: str = typer.Option(..., help="Run name to evaluate."),
    checkpoint: str = typer.Option("latest"),
    cache: Path = typer.Option(Path("data/cache/d1")),
    probe_size: int = typer.Option(8, help="Held-out meshes to evaluate on."),
    level: str = typer.Option("c1", help="Corruption applied to the input."),
    unshared: bool = typer.Option(False, help="Set if the run was trained unshared."),
) -> None:
    """Evaluate a checkpoint over the probe set, tracing every metric by T."""
    from flock.data.store import DatasetRepository
    from flock.domain.dynamics.state import CellConfig
    from flock.domain.skinning.corruption import CorruptionLevel, corrupt
    from flock.evaluation.probe import EVAL_ITERATIONS, evaluate_by_iteration
    from flock.evaluation.replication import load_checkpoint
    from flock.experiment.repository import RunRepository

    runs = RunRepository(Path("runs"))
    cell = CellConfig(shared_weights=not unshared)
    params = load_checkpoint(runs.path_for(run), cell)

    repo = DatasetRepository(cache)
    ids = repo.list_ids()[-probe_size:]          # held out from the training prefix
    samples = [repo.load(i)[0] for i in ids]
    rng = np.random.default_rng(0)
    initial = np.stack([
        corrupt(s.ground_truth, CorruptionLevel(level), s.corruption_context(), rng)[0].values
        for s in samples
    ])

    curves = evaluate_by_iteration(params, samples, cell, initial)
    header = "  ".join(f"T={t:<7}" for t in EVAL_ITERATIONS)
    typer.echo(f"{'metric':<12} {header}")
    for name, curve in curves.items():
        row = "  ".join(f"{v:<9.4f}" for v in curve.values)
        typer.echo(f"{name:<12} {row}")

    # §8: every curve is serialised beside the checkpoint that produced it, so a
    # printed table is never the only record of a result.
    runs.save_curves(
        run,
        level,
        {"iterations": list(EVAL_ITERATIONS), "meshes": ids,
         **{name: list(curve.values) for name, curve in curves.items()}},
    )

    # Indicative readings only: G1 also requires beating B3, and G2/G3 have
    # their own protocols. Both land at M4.
    l1 = curves["weight_l1"].values
    at1, at8, at32 = l1[EVAL_ITERATIONS.index(1)], l1[EVAL_ITERATIONS.index(8)], l1[-1]
    gain, drift = at8 / max(at1, 1e-9), at32 / max(at8, 1e-9)
    typer.echo(f"\nrecurrence gain  L1(T=8)/L1(T=1) = {gain:.3f}  (G1 wants <= 0.80)")
    typer.echo(f"drift past T=8   L1(T=32)/L1(T=8) = {drift:.3f}  (G2 wants <= 1.05)")
    typer.secho(
        "indicative only — G1 needs B3, G2/G3 need their own runs (M4)",
        fg=typer.colors.YELLOW,
    )


@app.command()
def repair(
    run: str = typer.Option(..., help="Run name to evaluate."),
    cache: Path = typer.Option(Path("data/cache/d1")),
    probe_size: int = typer.Option(8, help="Held-out meshes to evaluate on."),
    level: str = typer.Option("c1", help="Corruption applied to the converged state."),
    settle: int = typer.Option(8, help="Steps run before the damage."),
    seed: int = typer.Option(0),
    unshared: bool = typer.Option(False, help="Set if the run was trained unshared."),
) -> None:
    """Gate G3: does a converged state repair local damage without collateral harm?"""
    from flock.data.store import DatasetRepository
    from flock.domain.dynamics.state import CellConfig
    from flock.evaluation.gates import GateStatus, evaluate_gate
    from flock.evaluation.probe import evaluate_repair
    from flock.evaluation.replication import load_checkpoint
    from flock.experiment.repository import RunRepository

    runs = RunRepository(Path("runs"))
    cell = CellConfig(shared_weights=not unshared)
    params = load_checkpoint(runs.path_for(run), cell)

    repo = DatasetRepository(cache)
    ids = repo.list_ids()[-probe_size:]          # held out from the training prefix
    samples = [repo.load(i)[0] for i in ids]

    result = evaluate_repair(params, samples, cell, level=level, settle=settle, seed=seed)
    typer.echo(f"episodes          : {result['episodes']} ({result['dropped']} induced no error)")
    typer.echo(f"region size       : {result['region_fraction']:.1%} of real vertices")
    header = "  ".join(f"T={t:<5}" for t in result["iterations"])
    typer.echo(f"\n{'metric':<20} {header}")
    for name, curve in result["curves"].items():
        typer.echo(f"{name:<20} " + "  ".join(f"{v:<7.3f}" for v in curve))

    verdict = evaluate_gate("G3", {
        "repaired_fraction": result["repaired_fraction"],
        "collateral_relative": result["collateral_relative"],
    })
    typer.echo(
        f"\nrepaired @T={result['iterations'][-1]}  : "
        f"{result['repaired_fraction']:.3f}  (G3 needs >= 0.80)"
    )
    typer.echo(
        f"collateral        : {result['collateral_relative']:.3f}  (G3 needs < 0.10)"
    )
    runs.save_curves(run, f"repair_{level}", result)
    typer.secho(
        f"GATE G3: {verdict.status.value.upper()}",
        fg=typer.colors.GREEN if verdict.status is GateStatus.PASS else typer.colors.RED,
        bold=True,
    )


@app.command()
def ablate(
    ablation: str = typer.Option(..., help="Ablation key, e.g. a1."),
    base_run: str = typer.Option(...),
    seeds: int = typer.Option(3, help="Three seeds before any conclusion (spec §8)."),
) -> None:
    """Run an ablation's arms."""
    _pending("M4", "flock ablate")


@app.command()
def gates(run: str | None = typer.Option(None, help="Run to evaluate gates against.")) -> None:
    """Print the gate table, with status if a run is given."""
    if run is None:
        for gate in GATES:
            typer.echo(f"{gate.key:<5} {gate.question}")
            typer.echo(f"      criterion: {gate.criterion}")
        return
    _pending("M1 (G0) / M4 (G1-G4)", "flock gates <run>")


if __name__ == "__main__":
    app()
