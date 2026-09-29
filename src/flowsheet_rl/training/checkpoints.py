from dataclasses import asdict
from pathlib import Path

import torch

from ..actions import UNIT_NAMES
from ..config import ModelConfig
from ..graph_state import EDGE_DIM, GLOBAL_DIM, NODE_DIM
from ..models import ActorCritic

SCHEMA = {
    "node_dim": NODE_DIM,
    "edge_dim": EDGE_DIM,
    "global_dim": GLOBAL_DIM,
    "units": list(UNIT_NAMES),
    "parameter_coordinates": "normalized_0_1",
}


def save_checkpoint(path, model, config=None, optimizer=None, steps=0, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "schema": SCHEMA,
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "steps": steps,
        "experiment_config": asdict(config) if config else None,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("schema") != SCHEMA:
        raise ValueError("incompatible_checkpoint_schema")
    return payload


def load_checkpoint(path):
    payload = read_checkpoint(path)
    model = ActorCritic(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def transfer_weights(path, target, strategy="full"):
    if strategy not in ("full", "encoder"):
        raise ValueError("transfer strategy must be full or encoder")
    source = read_checkpoint(path)["state_dict"]
    target_state = target.state_dict()
    selected = {k: v for k, v in source.items() if strategy == "full" or k.startswith("encoder.")}
    incompatible_encoder = [
        k
        for k, v in selected.items()
        if k.startswith("encoder.") and (k not in target_state or target_state[k].shape != v.shape)
    ]
    if incompatible_encoder:
        raise ValueError(f"incompatible encoder dimensions: {incompatible_encoder}")
    loaded = {
        k: v for k, v in selected.items() if k in target_state and target_state[k].shape == v.shape
    }
    target.load_state_dict(loaded, strict=False)
    return {
        "strategy": strategy,
        "loaded": sorted(loaded),
        "reinitialized": sorted(set(target_state) - set(loaded)),
    }
