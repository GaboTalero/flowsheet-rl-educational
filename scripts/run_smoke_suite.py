"""Run bounded baseline, PPO/checkpoint and transfer checks into a new directory."""

import argparse
import json
from pathlib import Path

from flowsheet_rl.cli import main
from flowsheet_rl.config import load_config
from flowsheet_rl.evaluation import begin_experiment, write_json
from flowsheet_rl.visualization import plot_comparison


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(config_dir / "train.yaml")
    out = begin_experiment(args.out, config, "run_smoke_suite")
    main(["baselines", "--config", str(config_dir / "train.yaml"), "--out", str(out / "baselines")])
    main(["train", "--config", str(config_dir / "train.yaml"), "--out", str(out / "ppo")])
    main(
        [
            "evaluate",
            "--checkpoint",
            str(out / "ppo" / "checkpoint.pt"),
            "--out",
            str(out / "checkpoint_evaluation"),
        ]
    )
    main(
        ["transfer", "--config", str(config_dir / "transfer.yaml"), "--out", str(out / "transfer")]
    )
    summaries = json.loads((out / "baselines" / "comparison.json").read_text())
    summaries.append(json.loads((out / "ppo" / "evaluation" / "summary.json").read_text()))
    write_json(out / "comparison.json", summaries)
    plot_comparison(summaries, out / "comparison.png")
    print(f"Saved complete smoke records to {out.resolve()}")


if __name__ == "__main__":
    run()
