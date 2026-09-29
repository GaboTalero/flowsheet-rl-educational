"""Plot learning and engineering results from one completed training run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, values.mean() if len(values) else np.nan, dtype=float)
    result = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed training directory")
    parser.add_argument("--out", type=Path, help="Plot directory; defaults to RUN/plots")
    parser.add_argument("--window", type=int, default=250, help="Training-episode rolling window")
    args = parser.parse_args()

    run = args.run.resolve()
    out = (args.out or run / "plots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    training = read_json(run / "training.json")
    config = training["config"]
    episodes = training["episodes"]
    updates = training["updates"]
    curve = training["learning_curve"]
    if not episodes or not updates or not curve:
        raise ValueError("training.json does not contain completed episodes, updates, and evaluations")

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    eval_steps = np.asarray([point["training_steps"] for point in curve])
    eval_returns = np.asarray([point["return_mean"] for point in curve])
    eval_return_std = np.asarray([point["return_std"] for point in curve])
    axes[0, 0].plot(eval_steps, eval_returns, color="#246a8d", marker="o", markersize=3)
    axes[0, 0].fill_between(
        eval_steps,
        eval_returns - eval_return_std,
        eval_returns + eval_return_std,
        color="#246a8d",
        alpha=0.18,
        label="mean ± SD across evaluation feeds",
    )
    axes[0, 0].set_ylabel("Deterministic evaluation return")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        eval_steps, [point["feasible_rate"] for point in curve], marker="o", label="Feasible"
    )
    axes[0, 1].plot(
        eval_steps,
        [point["product_purity_mean"] for point in curve],
        label="Mean purity",
    )
    axes[0, 1].plot(
        eval_steps,
        [point["product_recovery_mean"] for point in curve],
        label="Mean recovery",
    )
    axes[0, 1].axhline(
        config["environment"]["purity_min"], color="#c44e52", linestyle="--", label="Purity limit"
    )
    axes[0, 1].axhline(
        config["environment"]["recovery_min"],
        color="#8172b3",
        linestyle=":",
        label="Recovery limit",
    )
    axes[0, 1].set_ylim(-0.03, 1.05)
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].legend(fontsize=8, ncol=2)

    train_steps = np.asarray([row["steps"] for row in episodes])
    train_returns = np.asarray([row["return"] for row in episodes])
    train_feasible = np.asarray([row["feasible"] for row in episodes], dtype=float)
    window = min(max(args.window, 1), len(episodes))
    axes[1, 0].plot(
        train_steps,
        rolling_mean(train_returns, window),
        color="#2a788e",
        label=f"Return, rolling {window} episodes",
    )
    feasible_axis = axes[1, 0].twinx()
    feasible_axis.plot(
        train_steps,
        rolling_mean(train_feasible, window),
        color="#55a868",
        alpha=0.8,
        label="Training feasible fraction",
    )
    axes[1, 0].set_ylabel("Stochastic training return")
    feasible_axis.set_ylabel("Feasible fraction")
    feasible_axis.set_ylim(-0.03, 1.03)
    lines = axes[1, 0].lines + feasible_axis.lines
    axes[1, 0].legend(lines, [line.get_label() for line in lines], fontsize=8)

    update_steps = [row["steps"] for row in updates]
    axes[1, 1].plot(update_steps, [row["loss"] for row in updates], label="PPO loss")
    diagnostic_axis = axes[1, 1].twinx()
    diagnostic_axis.plot(
        update_steps,
        [row["approx_kl"] for row in updates],
        color="#c44e52",
        alpha=0.75,
        label="Approximate KL",
    )
    diagnostic_axis.plot(
        update_steps,
        [row["clip_fraction"] for row in updates],
        color="#8172b3",
        alpha=0.65,
        label="Clip fraction",
    )
    axes[1, 1].set_ylabel("Combined PPO loss")
    diagnostic_axis.set_ylabel("KL / clipped fraction")
    lines = axes[1, 1].lines + diagnostic_axis.lines
    axes[1, 1].legend(lines, [line.get_label() for line in lines], fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Collected training transitions")
        axis.grid(alpha=0.2)
    summary = training["summary"]
    figure.suptitle(
        f"Training results: {summary['steps']:,} transitions, "
        f"{summary['episodes']:,} episodes, seed {config['train']['seed']}"
    )
    figure.tight_layout()
    overview = out / "training_overview.png"
    figure.savefig(overview, dpi=180, bbox_inches="tight")
    plt.close(figure)

    evaluation_csv = run / "evaluation" / "episodes.csv"
    if evaluation_csv.exists():
        with evaluation_csv.open(newline="", encoding="utf-8") as handle:
            evaluation = list(csv.DictReader(handle))
        seeds = [row["seed"] for row in evaluation]
        returns = [float(row["return"]) for row in evaluation]
        colors = ["#55a868" if row["feasible"] == "True" else "#c44e52" for row in evaluation]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].bar(seeds, returns, color=colors)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_ylabel("Return")
        axes[0].set_xlabel("Evaluation feed seed")
        axes[0].set_title("Green = feasible; red = infeasible")
        axes[1].plot(
            seeds,
            [float(row["product_purity"]) for row in evaluation],
            marker="o",
            label="Purity",
        )
        axes[1].plot(
            seeds,
            [float(row["product_recovery"]) for row in evaluation],
            marker="o",
            label="Recovery",
        )
        axes[1].axhline(config["environment"]["purity_min"], color="#c44e52", linestyle="--")
        axes[1].axhline(
            config["environment"]["recovery_min"], color="#8172b3", linestyle=":"
        )
        axes[1].set_ylim(0, 1.03)
        axes[1].set_ylabel("Fraction")
        axes[1].set_xlabel("Evaluation feed seed")
        axes[1].legend()
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("Final deterministic evaluation")
        figure.tight_layout()
        figure.savefig(out / "evaluation_by_seed.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    final = curve[-1]
    print(
        json.dumps(
            {
                "run": str(run),
                "plots": str(out),
                "training_steps": summary["steps"],
                "first_feasible_step": summary["first_feasible_step"],
                "final_evaluation_return_mean": final["return_mean"],
                "final_evaluation_feasible_rate": final["feasible_rate"],
                "final_evaluation_purity_mean": final["product_purity_mean"],
                "final_evaluation_recovery_mean": final["product_recovery_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
