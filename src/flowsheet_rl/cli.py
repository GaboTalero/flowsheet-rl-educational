"""Command line entry points. Completed experiment directories are never overwritten."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .config import (
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    RewardConfig,
    TrainConfig,
    load_config,
)
from .evaluation import begin_experiment, evaluate_policy, write_json
from .models import ActorCritic
from .training.checkpoints import load_checkpoint, save_checkpoint
from .training.ppo import seed_everything, train
from .transfer import run_transfer
from .visualization import plot_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Educational graph flowsheet RL")
    parser.add_argument(
        "command", choices=["inspect", "baselines", "train", "evaluate", "transfer"]
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--out", required=True, help="New experiment directory; must not already exist"
    )
    parser.add_argument("--checkpoint")
    starts = parser.add_mutually_exclusive_group()
    starts.add_argument(
        "--resume", help="Continue a new-format training checkpoint in a NEW directory"
    )
    starts.add_argument(
        "--initialize-from", help="Start a new run with weights only, including legacy checkpoints"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--fidelity", choices=["source", "target"])
    args = parser.parse_args(argv)
    transfer_settings = {}
    if args.command == "transfer" and args.config:
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        transfer_settings = raw.pop("transfer", {})
        constructors = {
            "environment": EnvironmentConfig,
            "model": ModelConfig,
            "reward": RewardConfig,
            "train": TrainConfig,
        }
        config = ExperimentConfig(**{key: constructors[key](**value) for key, value in raw.items()})
    else:
        config = load_config(args.config)
    model = None
    resume_payload = None
    if args.resume or args.initialize_from:
        if args.command != "train":
            parser.error("--resume and --initialize-from apply only to train")
        model, resume_payload = load_checkpoint(args.resume or args.initialize_from)
        if args.resume and "training_state" not in resume_payload:
            parser.error(
                "Legacy checkpoint: use --initialize-from; exact continuation was not saved."
            )
        if not args.config and resume_payload.get("experiment_config"):
            raw = resume_payload["experiment_config"]
            config = ExperimentConfig(
                EnvironmentConfig(**raw["environment"]),
                RewardConfig(**raw["reward"]),
                ModelConfig(**raw["model"]),
                TrainConfig(**raw["train"]),
            )
        if args.resume:
            model = None  # train() restores optimizer, collector and random streams too.
    if args.seeds is None:
        args.seeds = (
            resume_payload["training_state"]["evaluation_seeds"]
            if args.resume
            else [101, 102, 103, 104, 105]
        )
    if args.command == "evaluate":
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        model, payload = load_checkpoint(args.checkpoint)
        if not args.config and payload.get("experiment_config"):
            raw = payload["experiment_config"]
            config = ExperimentConfig(
                EnvironmentConfig(**raw["environment"]),
                RewardConfig(**raw["reward"]),
                ModelConfig(**raw["model"]),
                TrainConfig(**raw["train"]),
            )
    if args.seed is not None:
        config.train.seed = args.seed
    if args.updates is not None:
        config.train.updates = args.updates
    if args.rollout_steps is not None:
        config.train.rollout_steps = args.rollout_steps
    if args.fidelity:
        config.environment.fidelity = args.fidelity
    # Revalidate CLI overrides.
    config = ExperimentConfig(
        EnvironmentConfig(**asdict(config.environment)),
        RewardConfig(**asdict(config.reward)),
        ModelConfig(**asdict(config.model)),
        TrainConfig(**asdict(config.train)),
    )
    seed_everything(config.train.seed)
    output = begin_experiment(args.out, config, " ".join(argv or sys.argv[1:]))
    if args.command in ("inspect", "baselines"):
        summaries = []
        policies = (
            ["rule"] if args.command == "inspect" else ["random", "rule", ActorCritic(config.model)]
        )
        for policy in policies:
            name = policy if isinstance(policy, str) else "untrained"
            if name == "untrained":
                save_checkpoint(output / "untrained.pt", policy, config)
            _, summary = evaluate_policy(policy, config, args.seeds, name, output / name)
            summaries.append(summary)
        write_json(output / "comparison.json", summaries)
        plot_comparison(summaries, output / "comparison.png")
        print(
            json.dumps(
                [
                    {
                        "policy": s["policy"],
                        "return": s["return_mean"],
                        "feasible_rate": s["feasible_rate"],
                    }
                    for s in summaries
                ],
                indent=2,
            )
        )
    elif args.command == "train":
        try:
            model, record = train(
                config,
                output,
                model=model,
                evaluation_seeds=args.seeds,
                resume=args.resume,
                verbose=True,
            )
        except KeyboardInterrupt:
            print(
                f"Interrupted. Resume the last saved update from {output / 'checkpoint.pt'} "
                "in a new output directory.",
                flush=True,
            )
            raise SystemExit(130) from None
        evaluate_policy(model, config, args.seeds, "trained", output / "evaluation")
        print(json.dumps(record["summary"], indent=2))
    elif args.command == "evaluate":
        _, summary = evaluate_policy(model, config, args.seeds, "checkpoint", output / "evaluation")
        print(json.dumps(summary, indent=2))
    else:
        write_json(output / "transfer_settings.json", transfer_settings)
        result = run_transfer(config, transfer_settings, output)
        print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
