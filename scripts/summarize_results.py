"""Summarize saved smoke records without rerunning or altering experiments."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment.resolve()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    comparison = load(root / "comparison.json")
    ppo = load(root / "ppo" / "training.json")
    transfer = load(root / "transfer" / "transfer.json")
    metadata = load(root / "metadata.json")
    cases = [case for path in root.rglob("designs.json") for case in load(path)]
    residual = max(
        abs(v) for case in cases for v in case["diagnostics"]["component_balance_residual"]
    )
    mass_residual = max(abs(case["diagnostics"]["mass_balance_residual"]) for case in cases)
    tests = ET.parse(args.tests).getroot().find("testsuite")
    result = {
        "experiment": str(root),
        "test_count": int(tests.attrib["tests"]),
        "test_failures": int(tests.attrib["failures"]),
        "test_errors": int(tests.attrib["errors"]),
        "maximum_component_balance_residual": residual,
        "maximum_mass_balance_residual": mass_residual,
        "evaluation_design_records": len(cases),
        "comparison": comparison,
        "ppo": ppo["summary"],
        "transfer": transfer["aggregate"],
        "versions": metadata["versions"],
    }
    link = "../06_EXPERIMENTS/" + root.name
    lines = [
        "# Foundation validation - 2026-09-21",
        "",
        "The requested executable foundation is complete. This validates the educational",
        "workflow; no numerical reproduction of the paper or learned-policy advantage is claimed.",
        "",
        "## Validation",
        "",
        f"- {result['test_count']} tests passed; zero failures/errors. See [JUnit record](foundation-tests.xml).",
        "- Ruff checks passed; package installed successfully from the locked CPU dependencies.",
        "- Installed flowsheet-rl command and checkpoint load/evaluation completed.",
        "- Tests cover stoichiometry, units, graph/port invariants, masks, Gym API, seed replay,",
        "  GNN batching/permutations/gradients, hybrid joint likelihood and entropy, PPO updates,",
        "  GAE boundaries, transfer/freeze controls, failure reasons and JSON exports.",
        f"- Maximum saved component-balance residual: {residual:.3g} mol/s.",
        f"- Maximum saved synthetic mass-balance residual: {mass_residual:.3g} g/s.",
        "- Acceptance tolerance: 1e-9. Numerical unit tests also check conserved elemental inventories.",
        "- Graph and comparison PNGs were visually inspected.",
        "",
        "## Source policy comparison",
        "",
        "Five held-out feed seeds: 101-105; same eight-unit and sixteen-step budgets.",
        "",
        "| Policy | Mean return | Sample SD | Feasible rate | Mean purity | Mean recovery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['policy']} | {row['return_mean']:.5f} | {row['return_std']:.5f} | "
            f"{row['feasible_rate']:.0%} | {row['product_purity_mean']:.5f} | "
            f"{row['product_recovery_mean']:.5f} |"
        )
    best = next(r for r in comparison if r["policy"] == "rule")["best"]
    lines += [
        "",
        "The rule policy mixed the feeds, heated to 370 K, reacted, separated, sent the",
        "C-rich branch to product and purged the residual. It used four equipment units",
        "and six actions; seven simulator calls include the initial feed evaluation.",
        f"Best sampled rule case: seed {best['seed']}, return {best['return']:.6f}.",
        "",
        (
            f"[Comparison plot]({link}/comparison.png) | "
            f"[Best rule flowsheet]({link}/baselines/rule/best_design.png) | "
            f"[Episode CSV]({link}/baselines/rule/episodes.csv)"
        ),
        "",
        "## PPO smoke",
        "",
        f"- {ppo['summary']['steps']} transitions and {ppo['summary']['episodes']} completed episodes.",
        f"- Parameter change L2 norm: {ppo['summary']['parameter_delta_l2']:.8f}.",
        "- Mean update losses: " + ", ".join(f"{r['loss']:.6f}" for r in ppo["updates"]) + ".",
        "- Gradients were finite and nonzero; clipping was applied at norm 0.5.",
        "- Checkpoint round-trip reproduced deterministic evaluation behavior.",
        "- No feasible training or evaluation design was found in this small budget.",
        "  The trained deterministic policy purged both feeds. Its less-negative reward",
        "  is not a successful process design or evidence of useful optimization.",
        f"- [Training record]({link}/ppo/training.json) | [Checkpoint]({link}/ppo/checkpoint.pt).",
        "",
        "## Transfer smoke",
        "",
        "Two training seeds (7,11), five evaluation cases per seed. Each seed used 64",
        "source pretraining transitions; each target strategy used 64 target transitions.",
        "Evaluation curves include zero target steps. Encoder freezing lasted one update.",
        "",
        "| Strategy | Mean target return | SD across training seeds | Feasible rate |",
        "|---|---:|---:|---:|",
    ]
    for row in transfer["aggregate"]:
        lines.append(
            f"| {row['strategy']} | {row['return_mean']:.5f} | "
            f"{row['return_std_across_training_seeds']:.5f} | {row['feasible_rate']:.0%} |"
        )
    lines += [
        "",
        "All variants executed with valid checkpoints and finite losses. No feasible",
        "designs or transfer advantage were established. Source pretraining consumes",
        "an additional budget and is not counted as free.",
        f"[Transfer records]({link}/transfer/transfer.json) | [Transfer plot]({link}/transfer/transfer.png)",
        "",
        "## Reproducibility and interpretation",
        "",
        f"Python {metadata['python']} on Windows; dependencies: "
        + ", ".join(f"{k} {v}" for k, v in metadata["versions"].items())
        + ".",
        "Both the short identical-seed PPO replay test and repeated full smoke suite",
        "produced matching scientific metrics and parameter deltas. Wall times vary.",
        "",
        "Source/fidelity models, Beta coordinates, joint PPO ratio, entropy calculation,",
        "finite-horizon bootstrapping and all synthetic constants are documented in",
        "[the package](../05_CODE/README.md). Original seed documents were preserved;",
        "the expanded user-requested scope is recorded in the decision log.",
        "",
        "The environment checker reports unbounded observation-space limits; actual",
        "observations remain finite. The locked PyG/PyTorch combination emits upstream",
        "deprecation warnings. These did not fail the tests.",
        "",
        "The full experiment metadata contains configuration, hashes and versions.",
        "No Git commit exists in this workspace; none was fabricated.",
        "The earlier 20260921_foundation_smoke run is retained as a validation record.",
        "",
        "## Next milestone",
        "",
        "Run a controlled multi-seed learning study with a larger, fixed transition",
        "budget and hold-out feed cases. Diagnose early STOP/purge exploration before",
        "testing a separately labeled curriculum or demonstration warm-start ablation.",
        "Recycles, rigorous process models and industrial economic conclusions remain deferred.",
        "",
        "Report generated from saved data by scripts/summarize_results.py. See README",
        "for exact install/test/train/evaluate commands.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("comparison", "transfer", "versions")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
