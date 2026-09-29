import torch
from torch.distributions import Beta, Categorical


def masked_categorical(logits, mask):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("categorical_has_no_valid_action")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def beta_distribution(raw):
    concentrations = torch.nn.functional.softplus(raw) + 1.0
    return Beta(concentrations[..., 0], concentrations[..., 1])
