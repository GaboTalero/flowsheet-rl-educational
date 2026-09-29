import numpy as np
import torch

from flowsheet_rl.config import ExperimentConfig, TrainConfig
from flowsheet_rl.environment import FlowsheetEnv
from flowsheet_rl.models import ActorCritic
from flowsheet_rl.training.advantages import generalized_advantages
from flowsheet_rl.training.checkpoints import load_checkpoint, transfer_weights
from flowsheet_rl.training.ppo import seed_everything, train


def test_gae_cutoffs_and_episode_boundaries():
    adv, ret = generalized_advantages(
        [1, 2, 3],
        [0.5, 0.6, 0.7],
        [0.6, 50, 2],
        [True, False, True],
        [False, True, False],
        gamma=0.9,
        lam=0.8,
    )
    np.testing.assert_allclose(
        adv, [1 + 0.9 * 0.6 - 0.5 + 0.9 * 0.8 * 1.4, 1.4, 3 + 0.9 * 2 - 0.7], rtol=1e-6
    )
    np.testing.assert_allclose(ret, adv + [0.5, 0.6, 0.7], rtol=1e-6)


def test_ppo_update_checkpoint_reproducibility_and_transfer(tmp_path):
    config = ExperimentConfig(
        train=TrainConfig(updates=1, rollout_steps=16, epochs=1, minibatch_size=8)
    )
    model, record = train(config, tmp_path)
    assert record["summary"]["steps"] == 16
    assert record["summary"]["parameter_delta_l2"] > 0
    assert record["updates"][0]["gradient_norm"] > 0
    assert np.isfinite(record["updates"][0]["loss"])
    restored, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    obs, _ = FlowsheetEnv().reset(seed=0)
    assert model.act(obs) == restored.act(obs)
    assert payload["steps"] == 16
    other, repeat = train(config)
    assert record["summary"]["parameter_delta_l2"] == repeat["summary"]["parameter_delta_l2"]
    for key in model.state_dict():
        torch.testing.assert_close(model.state_dict()[key], other.state_dict()[key], rtol=0, atol=0)
    target = ActorCritic()
    report = transfer_weights(tmp_path / "checkpoint.pt", target, "encoder")
    assert report["loaded"] and report["reinitialized"]
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, target.encoder.state_dict()[key])
    target.freeze_encoder()
    assert all(not p.requires_grad for p in target.encoder.parameters())
    target.freeze_encoder(False)
    assert all(p.requires_grad for p in target.encoder.parameters())


def test_freeze_then_unfreeze(tmp_path):
    seed_everything(2)
    model = ActorCritic()
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    cfg = ExperimentConfig(
        train=TrainConfig(updates=1, rollout_steps=8, epochs=1, freeze_encoder_updates=1)
    )
    train(cfg, model=model)
    for k, v in before.items():
        torch.testing.assert_close(model.encoder.state_dict()[k], v, rtol=0, atol=0)
    cfg.train.freeze_encoder_updates = 0
    train(cfg, model=model)
    assert any(not torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items())


def test_incompatible_heads_are_reinitialized_and_schema_is_checked(tmp_path):
    import pytest

    from flowsheet_rl.training.checkpoints import save_checkpoint

    source = ActorCritic()
    path = tmp_path / "source.pt"
    save_checkpoint(path, source)
    target = ActorCritic()
    target.unit_head[-1] = torch.nn.Linear(target.config.hidden_dim, 7)
    before = target.unit_head[-1].weight.detach().clone()
    report = transfer_weights(path, target, "full")
    assert "unit_head.2.weight" in report["reinitialized"]
    torch.testing.assert_close(before, target.unit_head[-1].weight)
    payload = torch.load(path, weights_only=True)
    payload["schema"]["units"] = ["different"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path)
