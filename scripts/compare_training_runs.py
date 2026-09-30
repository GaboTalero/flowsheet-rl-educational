"""Compare completed CPU/GPU runs and create a compact benchmark report."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(spec):
    if "=" not in spec:
        raise ValueError("Each --run must have the form LABEL=PATH")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).resolve()
    training = read_json(path / "training.json")
    evaluation = read_json(path / "evaluation" / "summary.json")
    metadata_path = path / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    config = training["config"]
    summary = training["summary"]
    runtime = training.get("runtime", {})
    elapsed = float(summary["elapsed_seconds"])
    steps = int(summary["steps"])
    torch_version = runtime.get("torch_version") or metadata.get("versions", {}).get("torch")
    device = runtime.get("device", "cpu (legacy default)")
    accelerator = runtime.get("accelerator", "CPU" if device.startswith("cpu") else "unknown")
    return {
        "label": label,
        "path": str(path),
        "device": device,
        "accelerator": accelerator,
        "torch_version": torch_version,
        "fidelity": config["environment"]["fidelity"],
        "seed": config["train"]["seed"],
        "updates": config["train"]["updates"],
        "rollout_steps": config["train"]["rollout_steps"],
        "epochs": config["train"]["epochs"],
        "minibatch_size": config["train"]["minibatch_size"],
        "learning_rate": config["train"]["learning_rate"],
        "hidden_dim": config["model"]["hidden_dim"],
        "gnn_layers": config["model"]["layers"],
        "steps": steps,
        "episodes": summary["episodes"],
        "elapsed_seconds": elapsed,
        "transitions_per_second": steps / elapsed if elapsed else None,
        "first_feasible_step": summary.get("first_feasible_step"),
        "return_mean": evaluation["return_mean"],
        "return_std": evaluation["return_std"],
        "feasible_rate": evaluation["feasible_rate"],
        "purity_mean": evaluation["product_purity_mean"],
        "recovery_mean": evaluation["product_recovery_mean"],
        "energy_mean": evaluation["energy_mean"],
        "capital_cost_mean": evaluation["capital_cost_mean"],
        "num_units_mean": evaluation["num_units_mean"],
        "best_design": str(path / "evaluation" / "best_design.png"),
        "training_plot": str(path / "plots" / "training_overview.png"),
        "evaluation_plot": str(path / "plots" / "evaluation_by_seed.png"),
    }


def write_table(rows, out):
    keys = list(rows[0])
    with (out / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out / "benchmark_summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )


def plot(rows, out):
    labels = [row["label"] for row in rows]
    colors = ["#246a8d" if str(row["device"]).startswith("cpu") else "#dd8452" for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(labels, [r["elapsed_seconds"] / 60 for r in rows], color=colors)
    axes[0, 0].set_ylabel("Elapsed time (minutes)")
    axes[0, 0].set_title("Wall-clock training time")
    axes[0, 1].bar(labels, [r["transitions_per_second"] for r in rows], color=colors)
    axes[0, 1].set_ylabel("Transitions / second")
    axes[0, 1].set_title("Training throughput")
    axes[1, 0].bar(labels, [r["return_mean"] for r in rows], color=colors)
    axes[1, 0].set_ylabel("Mean deterministic return")
    axes[1, 0].set_title("Final evaluation return")
    width = 0.24
    positions = list(range(len(rows)))
    axes[1, 1].bar(
        [x - width for x in positions], [r["feasible_rate"] for r in rows], width, label="Feasible"
    )
    axes[1, 1].bar(positions, [r["purity_mean"] for r in rows], width, label="Purity")
    axes[1, 1].bar(
        [x + width for x in positions], [r["recovery_mean"] for r in rows], width, label="Recovery"
    )
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Final engineering metrics")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("RL-PSE CPU/GPU benchmark")
    figure.tight_layout()
    figure.savefig(out / "benchmark_overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_summary(rows):
    headings = [
        "label",
        "device",
        "accelerator",
        "fidelity",
        "steps",
        "elapsed_seconds",
        "transitions_per_second",
        "return_mean",
        "feasible_rate",
        "purity_mean",
        "recovery_mean",
    ]
    print(" | ".join(headings))
    print(" | ".join(["---"] * len(headings)))
    for row in rows:
        values = []
        for key in headings:
            value = row[key]
            values.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        print(" | ".join(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="Completed run as LABEL=PATH; repeat to compare"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [load_run(spec) for spec in args.run]
    args.out.mkdir(parents=True, exist_ok=True)
    write_table(rows, args.out)
    plot(rows, args.out)
    print_summary(rows)
    print(f"\nBenchmark files: {args.out.resolve()}")


if __name__ == "__main__":
    main()
