"""Matched target budgets with a separately reported source pretraining budget."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from .evaluation import evaluate_policy, write_json
from .models import ActorCritic
from .training.checkpoints import transfer_weights
from .training.ppo import seed_everything, train
from .visualization import plot_transfer


def run_transfer(config, settings, output):
    output = Path(output)
    settings = {
        "seeds": [7, 11],
        "evaluation_seeds": [101, 102, 103, 104, 105],
        "source_updates": 2,
        "target_updates": 2,
        "rollout_steps": 32,
        "freeze_encoder_updates": 1,
        "strategies": ["scratch", "full", "encoder", "encoder_frozen"],
        **settings,
    }
    runs = []
    source_records = []
    for seed in settings["seeds"]:
        source_config = deepcopy(config)
        source_config.environment.fidelity = "source"
        source_config.train.seed = seed
        source_config.train.updates = settings["source_updates"]
        source_config.train.rollout_steps = settings["rollout_steps"]
        source_config.train.freeze_encoder_updates = 0
        source_dir = output / f"seed_{seed}" / "source"
        _, source_record = train(
            source_config, source_dir, evaluation_seeds=settings["evaluation_seeds"]
        )
        source_records.append({"seed": seed, "training": source_record})
        for strategy in settings["strategies"]:
            if strategy not in ("scratch", "full", "encoder", "encoder_frozen"):
                raise ValueError(f"unknown transfer strategy: {strategy}")
            target_config = deepcopy(source_config)
            target_config.environment.fidelity = "target"
            target_config.train.updates = settings["target_updates"]
            target_config.train.freeze_encoder_updates = (
                settings["freeze_encoder_updates"] if strategy == "encoder_frozen" else 0
            )
            seed_everything(seed)
            target = ActorCritic(target_config.model)
            report = {
                "strategy": "scratch",
                "loaded": [],
                "reinitialized": list(target.state_dict()),
            }
            if strategy != "scratch":
                report = transfer_weights(
                    source_dir / "checkpoint.pt",
                    target,
                    "full" if strategy == "full" else "encoder",
                )
            run_dir = output / f"seed_{seed}" / strategy
            model, record = train(
                target_config, run_dir, target, evaluation_seeds=settings["evaluation_seeds"]
            )
            _, final = evaluate_policy(
                model,
                target_config,
                settings["evaluation_seeds"],
                strategy,
                output=run_dir / "evaluation",
            )
            runs.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "transfer": report,
                    "training": record,
                    "evaluation": final,
                }
            )
    aggregate = []
    for strategy in settings["strategies"]:
        subset = [r["evaluation"] for r in runs if r["strategy"] == strategy]
        returns = [r["return_mean"] for r in subset]
        aggregate.append(
            {
                "strategy": strategy,
                "training_seeds": len(subset),
                "return_mean": float(np.mean(returns)),
                "return_std_across_training_seeds": float(np.std(returns, ddof=1))
                if len(returns) > 1
                else 0.0,
                "feasible_rate": float(np.mean([s["feasible_rate"] for s in subset])),
            }
        )
    result = {
        "settings": settings,
        "source_runs": source_records,
        "target_runs": runs,
        "aggregate": aggregate,
        "interpretation": "Smoke validation only; no transfer advantage or convergence established.",
    }
    write_json(output / "transfer.json", result)
    plot_transfer(runs, output / "transfer.png")
    return result
