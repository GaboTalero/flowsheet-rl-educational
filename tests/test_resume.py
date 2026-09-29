"""Continuation must reproduce uninterrupted training, including partially built graphs."""

from copy import deepcopy

import pytest
import torch

from flowsheet_rl.config import ExperimentConfig, TrainConfig
from flowsheet_rl.models import ActorCritic
from flowsheet_rl.training.checkpoints import load_checkpoint, save_checkpoint
from flowsheet_rl.training.ppo import train


def test_exact_resume_across_rollout_boundary(tmp_path):
    cfg = ExperimentConfig(
        train=TrainConfig(updates=3, rollout_steps=7, epochs=1, minibatch_size=4)
    )
    uninterrupted, full = train(cfg, tmp_path / "full", evaluation_seeds=[101])
    first_cfg = deepcopy(cfg)
    first_cfg.train.updates = 1
    train(first_cfg, tmp_path / "first", evaluation_seeds=[101])
    resumed, split = train(cfg, tmp_path / "continued", resume=tmp_path / "first" / "checkpoint.pt")
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    assert full["episodes"] == split["episodes"]
    for a, b in zip(full["updates"], split["updates"]):
        for key in ("loss", "actor_loss", "critic_loss", "entropy", "approx_kl", "steps"):
            assert a[key] == b[key]
    _, payload = load_checkpoint(tmp_path / "continued" / "checkpoint.pt")
    assert payload["training_state"]["completed_updates"] == 3
    assert payload["steps"] == 21
    assert (tmp_path / "continued" / "progress.json").exists()


def test_resume_rejects_changed_science_and_legacy(tmp_path):
    cfg = ExperimentConfig(train=TrainConfig(updates=1, rollout_steps=4, epochs=1))
    train(cfg, tmp_path / "start")
    cfg.train.updates = 2
    cfg.environment.feed_a = 2
    with pytest.raises(ValueError, match="same_environment"):
        train(cfg, resume=tmp_path / "start" / "checkpoint.pt")
    save_checkpoint(tmp_path / "legacy.pt", ActorCritic(), cfg)
    with pytest.raises(ValueError, match="legacy"):
        train(cfg, resume=tmp_path / "legacy.pt")


def test_interrupted_run_keeps_last_complete_checkpoint(tmp_path, monkeypatch):
    from flowsheet_rl.training import ppo

    cfg = ExperimentConfig(train=TrainConfig(updates=3, rollout_steps=4, epochs=1))
    collect = ppo.Collector.collect
    calls = 0

    def interrupted(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return collect(self, *args)

    monkeypatch.setattr(ppo.Collector, "collect", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, tmp_path / "interrupted")
    _, payload = load_checkpoint(tmp_path / "interrupted" / "checkpoint.pt")
    assert payload["training_state"]["completed_updates"] == 1
    assert payload["steps"] == 4
