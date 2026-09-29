"""Identical seeded cases and design budgets for all policies."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .baselines import RandomValidPolicy, RuleBasedPolicy
from .config import config_hash
from .environment import FlowsheetEnv


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def begin_experiment(path, config, command):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    package_root = Path(__file__).resolve().parents[2]
    code_files = sorted(package_root.glob("src/**/*.py"))
    for directory in ("tests", "scripts", "configs"):
        code_files.extend(sorted(package_root.glob(f"{directory}/**/*.*")))
    code_files.extend(package_root / n for n in ("pyproject.toml", "uv.lock"))
    hashes = {
        str(f.relative_to(package_root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in code_files
        if f.is_file()
    }
    versions = {}
    for name in (
        "flowsheet-rl",
        "numpy",
        "torch",
        "torch-geometric",
        "gymnasium",
        "networkx",
        "matplotlib",
        "pytest",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(
        path / "metadata.json",
        {
            "command": command,
            "config": asdict(config),
            "config_sha256": config_hash(config),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": versions,
            "source_sha256": hashes,
            "revision": "workspace_has_no_git_repository",
        },
    )
    return path


def evaluate_policy(policy, config, seeds, name=None, output=None):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at_least_one_evaluation_seed_required")
    rows, designs = [], []
    start = time.perf_counter()
    first_time, first_episode = None, None
    name = name or (policy if isinstance(policy, str) else "neural")
    for episode, seed in enumerate(seeds):
        if isinstance(policy, str):
            if policy not in ("random", "rule"):
                raise ValueError("unknown_baseline")
            agent = RandomValidPolicy(seed) if policy == "random" else RuleBasedPolicy()
        else:
            agent = policy
            agent.eval()
        env = FlowsheetEnv(config.environment, config.reward)
        obs, _ = env.reset(seed=int(seed))
        total = 0.0
        episode_start = time.perf_counter()
        while not env.done:
            obs, reward, _, _, info = env.step(agent.act(obs))
            total += reward
        row = {
            "policy": name,
            "seed": int(seed),
            "return": total,
            **{
                k: info[k]
                for k in (
                    "feasible",
                    "topological_valid",
                    "physical_feasible",
                    "numerical_converged",
                    "specification_satisfied",
                    "closed",
                    "economically_attractive",
                    "product_purity",
                    "product_recovery",
                    "energy",
                    "capital_cost",
                    "profit_proxy",
                    "num_units",
                    "simulator_calls",
                    "steps",
                    "termination_reason",
                )
            },
            "runtime_seconds": time.perf_counter() - episode_start,
        }
        rows.append(row)
        designs.append(
            {
                "policy": name,
                "seed": int(seed),
                "graph": env.state.record(),
                "actions": env.history,
                "diagnostics": {k: v for k, v in info.items() if k != "action_masks"},
                "reward_breakdown": info["reward_breakdown"],
            }
        )
        if row["feasible"] and first_episode is None:
            first_episode = episode + 1
            first_time = time.perf_counter() - start
    summary = {
        "policy": name,
        "episodes": len(rows),
        "seeds": seeds,
        "feasible_rate": float(np.mean([r["feasible"] for r in rows])),
        "termination_reasons": dict(Counter(r["termination_reason"] for r in rows)),
        "episodes_to_first_feasible": first_episode,
        "seconds_to_first_feasible": first_time,
    }
    for metric in (
        "return",
        "product_purity",
        "product_recovery",
        "energy",
        "capital_cost",
        "profit_proxy",
        "num_units",
        "simulator_calls",
        "steps",
    ):
        values = [r[metric] for r in rows]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    best = max(range(len(rows)), key=lambda i: rows[i]["return"])
    summary["best"] = rows[best]
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "episodes.json", rows)
        write_json(output / "summary.json", summary)
        write_json(output / "designs.json", designs)
        write_json(output / "best_design.json", designs[best])
        with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        from .visualization import plot_flowsheet

        plot_flowsheet(designs[best]["graph"], output / "best_design.png")
    return rows, summary
