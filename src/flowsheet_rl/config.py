"""Validated, serializable experiment configuration."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class EnvironmentConfig:
    fidelity: str = "source"
    max_units: int = 8
    max_steps: int = 16
    feed_a: float = 1.0
    feed_b: float = 1.0
    feed_variation: float = 0.1
    feed_temperature: float = 300.0
    purity_min: float = 0.9
    recovery_min: float = 0.6
    balance_tolerance: float = 1e-9

    def __post_init__(self):
        if self.fidelity not in ("source", "target"):
            raise ValueError("fidelity must be source or target")
        if self.max_units < 1 or self.max_steps < 1:
            raise ValueError("positive construction budgets required")
        values = asdict(self)
        if any(not math.isfinite(v) for v in values.values() if isinstance(v, (float, int))):
            raise ValueError("configuration must be finite")
        if min(self.feed_a, self.feed_b, self.balance_tolerance) <= 0:
            raise ValueError("positive feeds and tolerance required")
        if not 0 <= self.feed_variation < 1 or not 280 <= self.feed_temperature <= 450:
            raise ValueError("invalid feed conditions")
        if not all(0 < x <= 1 for x in (self.purity_min, self.recovery_min)):
            raise ValueError("specifications must lie in (0,1]")

    @property
    def max_nodes(self):
        return 4 + 2 * self.max_units


@dataclass
class RewardConfig:
    product_value: float = 10.0
    utility: float = 0.2
    capital: float = 0.3
    complexity: float = 0.05
    success: float = 2.0
    specification: float = 4.0
    open_stream: float = 2.0
    failure: float = 10.0
    step: float = 0.01

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("reward weights must be finite and nonnegative")


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    layers: int = 2

    def __post_init__(self):
        if self.hidden_dim < 4 or self.layers < 1:
            raise ValueError("invalid model dimensions")


@dataclass
class TrainConfig:
    seed: int = 7
    updates: int = 4
    rollout_steps: int = 64
    epochs: int = 2
    minibatch_size: int = 16
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    freeze_encoder_updates: int = 0
    checkpoint_interval: int = 1
    evaluation_interval: int = 1

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("training configuration must be finite")
        if not 0 < self.clip_ratio < 1 or min(self.entropy_coef, self.value_coef) < 0:
            raise ValueError("invalid PPO clipping or loss weights")
        if min(self.checkpoint_interval, self.evaluation_interval) < 1:
            raise ValueError("checkpoint/evaluation intervals must be positive")
        if self.freeze_encoder_updates < 0:
            raise ValueError("negative freeze budget")
        if min(self.updates, self.rollout_steps, self.epochs, self.minibatch_size) < 1:
            raise ValueError("training budgets must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid discount or GAE lambda")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("positive optimizer parameters required")


@dataclass
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path=None):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    raw = raw or {}
    constructors = {
        "environment": EnvironmentConfig,
        "reward": RewardConfig,
        "model": ModelConfig,
        "train": TrainConfig,
    }
    unknown = set(raw) - constructors.keys()
    if unknown:
        raise ValueError(f"unknown configuration sections: {unknown}")
    return ExperimentConfig(**{k: constructors[k](**v) for k, v in raw.items()})


def config_hash(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
