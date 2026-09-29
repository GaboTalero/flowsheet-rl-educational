"""Run and resume a sequential, multi-seed PPO training campaign."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT.parent / "06_EXPERIMENTS" / "long_campaigns"


def keep_system_awake(enabled: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous = 0x80000000
    es_system_required = 0x00000001
    state = es_continuous | es_system_required if enabled else es_continuous
    ctypes.windll.kernel32.SetThreadExecutionState(state)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def target_budget(config_path: Path, updates: int | None, rollout_steps: int | None) -> tuple[int, int]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    train = raw.get("train", {})
    resolved_updates = updates if updates is not None else int(train.get("updates", 4))
    resolved_rollout = (
        rollout_steps if rollout_steps is not None else int(train.get("rollout_steps", 64))
    )
    return resolved_updates, resolved_rollout


def next_segment(campaign: Path, seed: int, label: str) -> Path:
    base = f"seed_{seed:04d}_{label}"
    index = 1
    while (campaign / f"{base}_{index:02d}").exists():
        index += 1
    return campaign / f"{base}_{index:02d}"


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = log_path.parent.parent / ".mpl-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    printable = subprocess.list2cmdline(command)
    print(f"\n$ {printable}", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{utc_now()}] $ {printable}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        except KeyboardInterrupt:
            process.send_signal(2)
            process.wait()
            raise
        return process.wait()


def create_manifest(args: argparse.Namespace, campaign: Path, config: Path) -> dict:
    updates, rollout_steps = target_budget(config, args.updates, args.rollout_steps)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "created",
        "project": str(PROJECT),
        "python": sys.executable,
        "config": str(config),
        "target_updates": updates,
        "rollout_steps": rollout_steps,
        "transitions_per_seed": updates * rollout_steps,
        "training_seeds": args.training_seeds,
        "evaluation_seeds": args.evaluation_seeds,
        "runs": {str(seed): {"status": "pending", "segments": []} for seed in args.training_seeds},
        "baselines": {"status": "pending" if not args.skip_baselines else "skipped"},
    }


def training_command(
    manifest: dict,
    seed: int,
    output: Path,
    resume: Path | None,
) -> list[str]:
    command = [sys.executable, str(PROJECT / "scripts" / "train.py")]
    if resume is None:
        command += ["--config", manifest["config"], "--seed", str(seed)]
        if manifest.get("rollout_steps_override"):
            command += ["--rollout-steps", str(manifest["rollout_steps"])]
    else:
        command += ["--resume", str(resume)]
    command += ["--updates", str(manifest["target_updates"])]
    command += ["--seeds", *[str(value) for value in manifest["evaluation_seeds"]]]
    command += ["--out", str(output)]
    return command


def run_baselines(manifest: dict, campaign: Path) -> None:
    state = manifest["baselines"]
    if state["status"] in {"complete", "skipped"}:
        return
    output = campaign / "baselines"
    if output.exists():
        state.update(status="failed", reason="existing_incomplete_baseline_directory")
        return
    state.update(status="running", started_at=utc_now(), output=str(output))
    manifest["updated_at"] = utc_now()
    atomic_json(campaign / "campaign.json", manifest)
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "run_baselines.py"),
        "--config",
        manifest["config"],
        "--seeds",
        *[str(value) for value in manifest["evaluation_seeds"]],
        "--out",
        str(output),
    ]
    return_code = run_logged(command, campaign / "logs" / "baselines.log")
    state.update(
        status="complete" if return_code == 0 else "failed",
        return_code=return_code,
        finished_at=utc_now(),
    )
    atomic_json(campaign / "campaign.json", manifest)


def run_seed(manifest: dict, campaign: Path, seed: int) -> bool:
    run = manifest["runs"][str(seed)]
    if run["status"] == "complete":
        print(f"seed {seed}: already complete", flush=True)
        return True

    resume = None
    if run["segments"]:
        previous = Path(run["segments"][-1]["output"])
        if (previous / "evaluation" / "summary.json").exists():
            run.update(status="complete", final_output=str(previous))
            manifest["updated_at"] = utc_now()
            atomic_json(campaign / "campaign.json", manifest)
            print(f"seed {seed}: recovered completed output", flush=True)
            return True
        checkpoint = previous / "checkpoint.pt"
        if checkpoint.exists():
            resume = checkpoint
            output = next_segment(campaign, seed, "resume")
            mode = "resume"
        else:
            output = next_segment(campaign, seed, "retry")
            mode = "retry"
    else:
        output = campaign / f"seed_{seed:04d}"
        mode = "fresh"

    segment = {
        "mode": mode,
        "status": "running",
        "started_at": utc_now(),
        "output": str(output),
        "resumed_from": str(resume) if resume else None,
    }
    run["status"] = "running"
    run["segments"].append(segment)
    manifest.update(status="running", updated_at=utc_now())
    atomic_json(campaign / "campaign.json", manifest)

    command = training_command(manifest, seed, output, resume)
    log_name = f"seed_{seed:04d}_{len(run['segments']):02d}.log"
    try:
        return_code = run_logged(command, campaign / "logs" / log_name)
    except KeyboardInterrupt:
        segment.update(status="interrupted", finished_at=utc_now())
        run["status"] = "interrupted"
        manifest.update(status="interrupted", updated_at=utc_now())
        atomic_json(campaign / "campaign.json", manifest)
        raise

    success = return_code == 0 and (output / "evaluation" / "summary.json").exists()
    segment.update(
        status="complete" if success else "failed",
        return_code=return_code,
        finished_at=utc_now(),
    )
    run.update(status="complete" if success else "failed", final_output=str(output))
    manifest["updated_at"] = utc_now()
    atomic_json(campaign / "campaign.json", manifest)
    return success


def plot_learning_curves(manifest: dict, campaign: Path) -> Path | None:
    os.environ["MPLCONFIGDIR"] = str(campaign / ".mpl-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    plotted = False
    for seed in manifest["training_seeds"]:
        run = manifest["runs"][str(seed)]
        if run["status"] != "complete":
            continue
        record = load_json(Path(run["final_output"]) / "training.json")
        curve = record["learning_curve"]
        if not curve:
            continue
        steps = [point["training_steps"] for point in curve]
        axes[0].plot(steps, [point["return_mean"] for point in curve], marker="o", label=f"seed {seed}")
        axes[1].plot(
            steps,
            [point["feasible_rate"] for point in curve],
            marker="o",
            label=f"seed {seed}",
        )
        plotted = True
    comparison_path = campaign / "baselines" / "comparison.json"
    if comparison_path.exists():
        rule = next(
            (row for row in load_json(comparison_path) if row["policy"] == "rule"),
            None,
        )
        if rule:
            axes[0].axhline(rule["return_mean"], color="#64748b", linestyle="--", label="rule")
            axes[1].axhline(rule["feasible_rate"], color="#64748b", linestyle="--", label="rule")
    if not plotted:
        plt.close(figure)
        return None
    axes[0].set_ylabel("Evaluation mean return")
    axes[1].set_ylabel("Evaluation feasible rate")
    axes[1].set_xlabel("Training transitions")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Held-out evaluation during PPO training")
    figure.tight_layout()
    output = campaign / "learning_curves.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return output


def summarize(manifest: dict, campaign: Path) -> dict:
    rows = []
    for seed in manifest["training_seeds"]:
        run = manifest["runs"][str(seed)]
        if run["status"] != "complete":
            continue
        output = Path(run["final_output"])
        evaluation = load_json(output / "evaluation" / "summary.json")
        training = load_json(output / "training.json")["summary"]
        rows.append(
            {
                "training_seed": seed,
                "steps": training["steps"],
                "episodes": training["episodes"],
                "elapsed_seconds": training["elapsed_seconds"],
                "first_feasible_step": training["first_feasible_step"],
                "evaluation_feasible_rate": evaluation["feasible_rate"],
                "evaluation_return_mean": evaluation["return_mean"],
                "evaluation_return_std": evaluation["return_std"],
                "purity_mean": evaluation["product_purity_mean"],
                "recovery_mean": evaluation["product_recovery_mean"],
                "energy_mean": evaluation["energy_mean"],
                "capital_cost_mean": evaluation["capital_cost_mean"],
                "best_return": evaluation["best"]["return"],
                "best_feasible": evaluation["best"]["feasible"],
                "output": str(output),
            }
        )

    metrics = {
        "evaluation_feasible_rate": "feasible_rate",
        "evaluation_return_mean": "return",
        "purity_mean": "purity",
        "recovery_mean": "recovery",
        "energy_mean": "energy",
        "capital_cost_mean": "capital_cost",
    }
    aggregate = {
        "completed_training_seeds": len(rows),
        "requested_training_seeds": len(manifest["training_seeds"]),
    }
    for metric, label in metrics.items():
        values = [float(row[metric]) for row in rows]
        aggregate[f"{label}_mean_across_training_seeds"] = (
            float(np.mean(values)) if values else None
        )
        aggregate[f"{label}_std_across_training_seeds"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        )

    result = {
        "campaign": str(campaign),
        "generated_at": utc_now(),
        "budget": {
            "updates_per_seed": manifest["target_updates"],
            "rollout_steps": manifest["rollout_steps"],
            "transitions_per_seed": manifest["transitions_per_seed"],
        },
        "aggregate": aggregate,
        "runs": rows,
        "baseline_comparison": (
            load_json(campaign / "baselines" / "comparison.json")
            if (campaign / "baselines" / "comparison.json").exists()
            else None
        ),
    }
    atomic_json(campaign / "campaign_summary.json", result)
    if rows:
        with (campaign / "campaign_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_learning_curves(manifest, campaign)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "train_long.yaml")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--updates", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--name", help="New campaign folder name; defaults to a UTC timestamp")
    parser.add_argument("--resume-campaign", type=Path)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keep_system_awake(True)
    atexit.register(keep_system_awake, False)
    if args.resume_campaign:
        campaign = args.resume_campaign.resolve()
        manifest = load_json(campaign / "campaign.json")
    else:
        config = args.config.resolve()
        if not config.exists():
            raise FileNotFoundError(config)
        name = args.name or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        campaign = (args.out_root / name).resolve()
        campaign.mkdir(parents=True, exist_ok=False)
        manifest = create_manifest(args, campaign, config)
        manifest["rollout_steps_override"] = args.rollout_steps is not None
        atomic_json(campaign / "campaign.json", manifest)

    print(f"campaign: {campaign}", flush=True)
    run_baselines(manifest, campaign)
    failed = False
    for seed in manifest["training_seeds"]:
        success = run_seed(manifest, campaign, int(seed))
        failed = failed or not success
        if not success and not args.continue_on_error:
            break

    all_complete = all(run["status"] == "complete" for run in manifest["runs"].values())
    manifest.update(
        status="complete" if all_complete else "failed" if failed else "incomplete",
        updated_at=utc_now(),
    )
    atomic_json(campaign / "campaign.json", manifest)
    summary = summarize(manifest, campaign)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)
    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
