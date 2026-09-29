import json

from flowsheet_rl.config import ExperimentConfig
from flowsheet_rl.evaluation import begin_experiment, evaluate_policy
from flowsheet_rl.transfer import run_transfer


def test_results_and_machine_readable_outputs(tmp_path):
    config = ExperimentConfig()
    _, summary = evaluate_policy("rule", config, [101, 102], output=tmp_path / "rule")
    assert summary["feasible_rate"] == 1
    assert summary["return_std"] >= 0
    assert (tmp_path / "rule" / "episodes.csv").is_file()
    assert json.loads((tmp_path / "rule" / "best_design.json").read_text())["graph"]["nodes"]
    begin_experiment(tmp_path / "new", config, "test")
    import pytest

    with pytest.raises(FileExistsError):
        begin_experiment(tmp_path / "new", config, "test")


def test_minimal_transfer_comparison(tmp_path):
    config = ExperimentConfig()
    config.train.epochs = 1
    settings = {
        "seeds": [3],
        "evaluation_seeds": [100],
        "source_updates": 1,
        "target_updates": 2,
        "rollout_steps": 4,
        "strategies": ["scratch", "full", "encoder_frozen"],
    }
    result = run_transfer(config, settings, tmp_path)
    assert len(result["target_runs"]) == 3
    assert all(r["training"]["summary"]["steps"] == 8 for r in result["target_runs"])
    frozen = result["target_runs"][-1]["training"]["updates"]
    assert frozen[0]["encoder_frozen"] and not frozen[1]["encoder_frozen"]
    assert (tmp_path / "transfer.png").is_file()
