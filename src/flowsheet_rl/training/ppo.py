"""Joint-ratio PPO with periodic durable checkpoints and selectable Torch device."""

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..environment import FlowsheetEnv
from ..models import ActorCritic
from .checkpoints import load_checkpoint, save_checkpoint
from .rollout import Collector


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    # Some CUDA scatter kernels used by graph operations may only provide a
    # nondeterministic implementation. Keep strict determinism on CPU and warn
    # rather than abort on an accelerator.
    torch.use_deterministic_algorithms(True, warn_only=torch.cuda.is_available())


def resolve_device(requested="cpu"):
    requested = str(requested).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in ("cpu", "cuda"):
        raise ValueError("device_must_be_cpu_cuda_or_auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_requested_but_not_available")
    return torch.device(requested)


def _move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _atomic_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _validate_resume(config, payload):
    if "training_state" not in payload:
        raise ValueError(
            "legacy_checkpoint_has_no_resume_state; use --initialize-from for a new run"
        )
    old, new = payload["experiment_config"], asdict(config)
    from ..config import TrainConfig

    old = {**old, "train": asdict(TrainConfig(**old["train"]))}
    for entry in (old, new):
        entry["train"] = {
            k: v
            for k, v in entry["train"].items()
            if k not in ("updates", "checkpoint_interval", "evaluation_interval")
        }
    if old != new:
        raise ValueError("resume_requires_same_environment_model_reward_and_optimizer_settings")


def train(
    config,
    output=None,
    model=None,
    evaluation_seeds=None,
    resume=None,
    verbose=False,
    device="cpu",
):
    seed_everything(config.train.seed)
    requested_device = str(device)
    run_device = resolve_device(device)
    cfg = config.train
    payload = None
    if resume is not None:
        if model is not None:
            raise ValueError("cannot_resume_and_supply_model")
        model, payload = load_checkpoint(resume)
        _validate_resume(config, payload)
    model = (model or ActorCritic(config.model)).to(run_device)
    initial = torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()]).clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    collector = Collector(FlowsheetEnv(config.environment, config.reward), cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    history, episodes, learning_curve = [], [], []
    start_update, elapsed_before = 0, 0.0
    if payload:
        state = payload["training_state"]
        checkpoint_device = state.get("device_type", "cpu")
        if checkpoint_device != run_device.type:
            raise ValueError("resume_requires_same_device_type")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _move_optimizer(optimizer, run_device)
        collector.load_state_dict(state["collector"])
        rng.bit_generator.state = state["minibatch_rng"]
        torch.set_rng_state(state["torch_rng"])
        if run_device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        initial = state["initial_parameters"]
        old_record = state["record"]
        history, episodes, learning_curve = (
            old_record["updates"],
            old_record["episodes"],
            old_record["learning_curve"],
        )
        start_update = state["completed_updates"]
        elapsed_before = old_record["summary"]["elapsed_seconds"]
        saved_seeds = state["evaluation_seeds"]
        if evaluation_seeds is not None and list(evaluation_seeds) != saved_seeds:
            raise ValueError("resume_requires_same_evaluation_seeds")
        evaluation_seeds = saved_seeds
        if cfg.updates <= start_update:
            raise ValueError("updates_is_total_budget_and_must_exceed_completed_updates")
    elif evaluation_seeds is not None:
        evaluation_seeds = list(evaluation_seeds)
    if run_device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = Path(output) if output is not None else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        # Never reuse an existing checkpoint directory through the Python API.
        if (output / "checkpoint.pt").exists():
            raise FileExistsError(output / "checkpoint.pt")

    def measure(steps):
        if evaluation_seeds is not None:
            from ..evaluation import evaluate_policy

            summary = evaluate_policy(model, config, evaluation_seeds, "policy")[1]
            learning_curve.append({"training_steps": steps, **summary})

    def record():
        if run_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = elapsed_before + time.perf_counter() - start
        final = torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])
        return {
            "config": asdict(config),
            "runtime": {
                "requested_device": requested_device,
                "device": str(run_device),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "accelerator": (
                    torch.cuda.get_device_name(run_device) if run_device.type == "cuda" else "CPU"
                ),
            },
            "summary": {
                "steps": collector.total_steps,
                "episodes": collector.total_episodes,
                "parameter_delta_l2": float(torch.linalg.vector_norm(final - initial)),
                "first_feasible_step": collector.first_feasible_step,
                "first_feasible_episode": collector.first_feasible_episode,
                "elapsed_seconds": elapsed,
                "transitions_per_second": (
                    collector.total_steps / elapsed if elapsed > 0 else None
                ),
            },
            "updates": history,
            "episodes": episodes,
            "learning_curve": learning_curve,
            "resumed_from": str(Path(resume).resolve()) if resume else None,
        }

    def persist(completed):
        snapshot = record()
        if output:
            state = {
                "completed_updates": completed,
                "collector": collector.state_dict(),
                "minibatch_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if run_device.type == "cuda" else None,
                "device_type": run_device.type,
                "initial_parameters": initial,
                "record": snapshot,
                "evaluation_seeds": evaluation_seeds,
            }
            save_checkpoint(
                output / "checkpoint.pt",
                model,
                config,
                optimizer,
                collector.total_steps,
                training_state=state,
            )
            _atomic_json(output / "training.json", snapshot)
        return snapshot

    if payload is None:
        measure(0)
    persist(start_update)  # A valid recovery point exists even before the first update.
    for update in range(start_update, cfg.updates):
        model.freeze_encoder(update < cfg.freeze_encoder_updates)
        rollout = collector.collect(model, cfg.rollout_steps, cfg.gamma, cfg.gae_lambda)
        episodes.extend(rollout.episodes)
        adv = torch.as_tensor(rollout.advantages, device=model.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        returns = torch.as_tensor(rollout.returns, device=model.device)
        old_logs = torch.tensor(rollout.log_probs, dtype=torch.float32, device=model.device)
        model.train()
        losses, norms, kls, clipped, entropies, actors, critics = [], [], [], [], [], [], []
        for _ in range(cfg.epochs):
            indices = rng.permutation(cfg.rollout_steps)
            for begin in range(0, cfg.rollout_steps, cfg.minibatch_size):
                batch = indices[begin : begin + cfg.minibatch_size]
                log_probs, entropy, values = model.evaluate_actions(
                    [rollout.observations[i] for i in batch], [rollout.actions[i] for i in batch]
                )
                ratio = torch.exp(log_probs - old_logs[batch])
                objective = torch.minimum(
                    ratio * adv[batch],
                    ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv[batch],
                )
                actor_loss = -objective.mean()
                critic_loss = (values - returns[batch]).square().mean()
                loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite_ppo_loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                with torch.no_grad():
                    losses.append(float(loss))
                    actors.append(float(actor_loss))
                    critics.append(float(critic_loss))
                    norms.append(float(norm))
                    entropies.append(float(entropy.mean()))
                    kls.append(float(((ratio - 1) - (log_probs - old_logs[batch])).mean()))
                    clipped.append(float(((ratio - 1).abs() > cfg.clip_ratio).float().mean()))
        entry = {
            "update": update + 1,
            "steps": collector.total_steps,
            "episodes": collector.total_episodes,
            "loss": float(np.mean(losses)),
            "actor_loss": float(np.mean(actors)),
            "critic_loss": float(np.mean(critics)),
            "gradient_norm": float(np.mean(norms)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_fraction": float(np.mean(clipped)),
            "encoder_frozen": update < cfg.freeze_encoder_updates,
            "elapsed_seconds": elapsed_before + time.perf_counter() - start,
        }
        history.append(entry)
        if (update + 1) % cfg.evaluation_interval == 0 or update + 1 == cfg.updates:
            measure(collector.total_steps)
        if (update + 1) % cfg.checkpoint_interval == 0 or update + 1 == cfg.updates:
            persist(update + 1)
        if output:
            _atomic_json(
                output / "progress.json",
                {
                    **entry,
                    "target_updates": cfg.updates,
                    "first_feasible_step": collector.first_feasible_step,
                },
            )
        if verbose:
            print(
                f"update {update + 1}/{cfg.updates} | steps={collector.total_steps} "
                f"| episodes={collector.total_episodes} | loss={entry['loss']:.4f} "
                f"| first_feasible_step={collector.first_feasible_step}",
                flush=True,
            )
    model.freeze_encoder(False)
    model.eval()
    return model, persist(cfg.updates)
