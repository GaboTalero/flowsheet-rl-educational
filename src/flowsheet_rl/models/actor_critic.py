"""Conditional hybrid policy. Joint entropy is an exact discrete-branch expectation."""

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.data import Batch

from ..actions import CONTINUOUS_UNITS, Action, Unit
from ..config import ModelConfig
from ..graph_state import to_pyg
from .distributions import beta_distribution, masked_categorical
from .gnn_encoder import GNNEncoder


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, outputs))


@dataclass
class PolicyOutput:
    action: Action
    component_log_probs: dict
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    masks: dict
    distribution_parameters: dict


class ActorCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        h = self.config.hidden_dim
        self.encoder = GNNEncoder(h, self.config.layers)
        self.stream_head = mlp(2 * h, 1, h)
        self.stop_head = mlp(h, 1, h)
        self.unit_head = mlp(2 * h, len(Unit), h)
        self.partner_head = mlp(3 * h, 1, h)
        self.parameter_heads = nn.ModuleDict(
            {str(int(unit)): mlp(2 * h, 2, h) for unit in CONTINUOUS_UNITS}
        )
        self.critic = mlp(h, 1, h)

    @property
    def device(self):
        return next(self.parameters()).device

    def value(self, observation):
        _, graph = self.encoder(to_pyg(observation).to(self.device))
        return self.critic(graph).squeeze()

    def _distributions(self, observation, nodes, graph):
        n = nodes.shape[0]
        contexts = torch.cat([nodes, graph.expand(n, -1)], dim=-1)
        stream_logits = torch.cat(
            [self.stream_head(contexts).squeeze(-1), self.stop_head(graph).reshape(1)]
        )
        mask = torch.as_tensor(observation["stream_mask"], device=self.device, dtype=torch.bool)
        stream = masked_categorical(stream_logits, torch.cat([mask[:n], mask[-1:]]))
        open_ids = mask[:n].nonzero().flatten().tolist()
        units, parameters, partners = {}, {}, {}
        for index in open_ids:
            units[index] = masked_categorical(
                self.unit_head(contexts[index]), observation["unit_mask"][index]
            )
            parameters[index] = {
                unit: beta_distribution(self.parameter_heads[str(int(unit))](contexts[index]))
                for unit in CONTINUOUS_UNITS
            }
            if bool(observation["unit_mask"][index, Unit.MIXER]):
                partner_context = torch.cat([contexts[index].expand(n, -1), nodes], dim=-1)
                partner_mask = mask[:n].clone()
                partner_mask[index] = False
                partners[index] = masked_categorical(
                    self.partner_head(partner_context).squeeze(-1), partner_mask
                )
        # Chain rule H(S,U,P)=H(S)+sum_s p(s)[H(U|s)+sum_u p(u|s)H(P|s,u)].
        # STOP has no descendants. Mixer has a discrete partner instead of a parameter.
        entropy = stream.entropy()
        for index in open_ids:
            conditional = units[index].entropy()
            for unit in CONTINUOUS_UNITS:
                conditional = (
                    conditional + units[index].probs[unit] * parameters[index][unit].entropy()
                )
            if index in partners:
                conditional = (
                    conditional + units[index].probs[Unit.MIXER] * partners[index].entropy()
                )
            entropy = entropy + stream.probs[index] * conditional
        return stream, units, parameters, partners, entropy

    def _output(self, observation, nodes, graph, action=None, deterministic=False):
        stream, units, params, partners, entropy = self._distributions(observation, nodes, graph)
        stop = len(observation["stream_mask"]) - 1
        n = nodes.shape[0]
        choose = lambda dist: dist.probs.argmax() if deterministic else dist.sample()
        if action is None:
            selected = int(choose(stream))
            if selected == n:
                action = Action(stop)
            else:
                unit = int(choose(units[selected]))
                parameter, partner = 0.0, 0
                if unit in CONTINUOUS_UNITS:
                    distribution = params[selected][unit]
                    parameter = float(distribution.mean if deterministic else distribution.sample())
                if unit == Unit.MIXER:
                    partner = int(choose(partners[selected]))
                action = Action(selected, unit, parameter, partner)
        selected = n if action.stream == stop else action.stream
        components = {"stream": stream.log_prob(torch.tensor(selected, device=self.device))}
        diagnostics = {"stream_probabilities": stream.probs}
        if selected != n:
            unit_distribution = units[selected]
            components["unit"] = unit_distribution.log_prob(
                torch.tensor(action.unit, device=self.device)
            )
            diagnostics["unit_probabilities"] = unit_distribution.probs
            if action.unit in CONTINUOUS_UNITS:
                if not 0 < action.parameter < 1:
                    raise ValueError("Beta likelihood requires an interior parameter")
                distribution = params[selected][action.unit]
                components["parameter"] = distribution.log_prob(
                    torch.tensor(action.parameter, device=self.device)
                )
                diagnostics.update(
                    alpha=distribution.concentration1, beta=distribution.concentration0
                )
            if action.unit == Unit.MIXER:
                components["partner"] = partners[selected].log_prob(
                    torch.tensor(action.partner, device=self.device)
                )
        log_prob = torch.stack(list(components.values())).sum()
        return PolicyOutput(
            action,
            components,
            log_prob,
            entropy,
            self.critic(graph).squeeze(),
            {k: observation[k] for k in ("stream_mask", "unit_mask")},
            diagnostics,
        )

    def forward(self, observation, action=None, deterministic=False):
        nodes, graphs = self.encoder(to_pyg(observation).to(self.device))
        return self._output(observation, nodes, graphs[0], action, deterministic)

    def evaluate_actions(self, observations, actions):
        batch = Batch.from_data_list([to_pyg(o) for o in observations]).to(self.device)
        nodes, graphs = self.encoder(batch)
        outputs = [
            self._output(o, nodes[batch.ptr[i] : batch.ptr[i + 1]], graphs[i], a)
            for i, (o, a) in enumerate(zip(observations, actions))
        ]
        return (
            torch.stack([o.log_prob for o in outputs]),
            torch.stack([o.entropy for o in outputs]),
            torch.stack([o.value for o in outputs]),
        )

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        return self(observation, deterministic=deterministic).action

    def freeze_encoder(self, freeze=True):
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not freeze)
