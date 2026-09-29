import numpy as np
import torch
from torch_geometric.data import Batch

from flowsheet_rl.actions import Action, Unit
from flowsheet_rl.environment import FlowsheetEnv
from flowsheet_rl.graph_state import to_pyg
from flowsheet_rl.models.gnn_encoder import GNNEncoder


def test_shapes_batch_permutations_gradients():
    torch.manual_seed(0)
    env = FlowsheetEnv()
    obs, _ = env.reset(seed=0)
    first = to_pyg(obs)
    obs, *_ = env.step(Action(1, Unit.MIXER, partner=3))
    second = to_pyg(obs)
    encoder = GNNEncoder(16, 2)
    h, g = encoder(second)
    assert h.shape == (5, 16) and g.shape == (1, 16)
    batch_h, batch_g = encoder(Batch.from_data_list([first, second]))
    assert batch_h.shape == (9, 16) and batch_g.shape == (2, 16)
    torch.testing.assert_close(batch_g[1], g[0], atol=1e-6, rtol=1e-6)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    inverse = torch.argsort(permutation)
    shuffled = second.clone()
    shuffled.x = second.x[permutation]
    shuffled.edge_index = inverse[second.edge_index]
    shuffled_h, shuffled_g = encoder(shuffled)
    torch.testing.assert_close(shuffled_h, h[permutation])
    torch.testing.assert_close(shuffled_g, g)
    (g.square().sum() + h.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert sum(float(p.grad.abs().sum()) for p in encoder.parameters()) > 0
    assert np.isfinite(batch_h.detach().numpy()).all()
