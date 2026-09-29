"""Dimensions: x[N,15], edge_attr[E,9], globals[B,9] -> h[N,H], g[B,H]."""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool

from ..graph_state import EDGE_DIM, GLOBAL_DIM, NODE_DIM


class DirectedLayer(MessagePassing):
    def __init__(self, hidden):
        super().__init__(aggr="add", flow="source_to_target")
        self.message_net = nn.Sequential(
            nn.Linear(hidden + EDGE_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.self_net = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, edge_attr):
        incoming = self.propagate(edge_index, x=h, edge_attr=edge_attr)
        return self.norm(h + torch.nn.functional.silu(self.self_net(h) + incoming))

    def message(self, x_j, edge_attr):
        return self.message_net(torch.cat([x_j, edge_attr], dim=-1))


class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim=32, layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input = nn.Linear(NODE_DIM, hidden_dim)
        self.layers = nn.ModuleList(DirectedLayer(hidden_dim) for _ in range(layers))
        self.readout = nn.Sequential(nn.Linear(hidden_dim + GLOBAL_DIM, hidden_dim), nn.Tanh())

    def forward(self, data):
        h = torch.nn.functional.silu(self.input(data.x))
        for layer in self.layers:
            h = layer(h, data.edge_index, data.edge_attr)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = global_add_pool(h, batch)
        return h, self.readout(torch.cat([pooled, data.globals], dim=-1))
