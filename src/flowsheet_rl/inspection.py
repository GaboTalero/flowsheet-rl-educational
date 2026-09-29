"""Small notebook/report helpers; no changes to the engineering or learning model."""

from dataclasses import asdict

import torch

from .graph_state import to_pyg


def stream_table(state):
    return [
        {
            "source": s,
            "target": t,
            "port": a["port"],
            "A": float(a["stream"].flow[0]),
            "B": float(a["stream"].flow[1]),
            "C": float(a["stream"].flow[2]),
            "total": a["stream"].total,
            "temperature_K": a["stream"].temperature,
            "pressure_bar": a["stream"].pressure,
        }
        for s, t, a in sorted(state.graph.edges(data=True))
    ]


@torch.no_grad()
def inspect_decision(model, observation, action=None):
    data = to_pyg(observation)
    h0 = torch.nn.functional.silu(model.encoder.input(data.x))
    layer = model.encoder.layers[0]
    messages = layer.message_net(torch.cat([h0[data.edge_index[0]], data.edge_attr], -1))
    aggregates = torch.zeros_like(h0).index_add_(0, data.edge_index[1], messages)
    h1 = layer.norm(h0 + torch.nn.functional.silu(layer.self_net(h0) + aggregates))
    nodes, graph = model.encoder(data)
    out = model(observation, action=action, deterministic=True)
    return {
        "action_origin": "provided_action_scored_by_policy"
        if action is not None
        else "policy_greedy",
        "action": asdict(out.action),
        "value": float(out.value),
        "component_log_probs": {k: float(v) for k, v in out.component_log_probs.items()},
        "joint_log_probability": float(out.log_prob),
        "entropy": float(out.entropy),
        "distributions": {
            k: v.detach().cpu().tolist() for k, v in out.distribution_parameters.items()
        },
        "shapes": {
            "nodes": list(data.x.shape),
            "edges": list(data.edge_attr.shape),
            "edge_index": list(data.edge_index.shape),
            "global_features": list(data.globals.shape),
            "node_embeddings": list(nodes.shape),
            "graph_embedding": list(graph.shape),
        },
        "X": data.x.tolist(),
        "Z": data.edge_attr.tolist(),
        "edge_index": data.edge_index.tolist(),
        "global_features": data.globals.tolist(),
        "stream_mask": observation["stream_mask"].tolist(),
        "unit_mask": observation["unit_mask"].tolist(),
        "message_example": {
            "edge": data.edge_index[:, 0].tolist(),
            "initial_source_first4": h0[0, :4].tolist(),
            "first_message_first4": messages[0, :4].tolist(),
            "receiver_updated_first4": h1[data.edge_index[1, 0], :4].tolist(),
            "message_norm": float(messages[0].norm()),
        },
        "graph_embedding_first4": graph[0, :4].tolist(),
    }


def trace_episode(env, policy, seed=103, scoring_model=None):
    obs, _ = env.reset(seed=seed)
    trace = []
    while not env.done:
        action = policy.act(obs)
        entry = {
            "step": len(trace),
            "before_graph": env.state.record(),
            "action": asdict(action),
            "streams_before": stream_table(env.state),
        }
        if scoring_model is not None:
            entry["neural_score"] = inspect_decision(scoring_model, obs, action)
        obs, reward, terminated, truncated, info = env.step(action)
        entry.update(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            diagnostics={k: v for k, v in info.items() if k != "action_masks"},
            after_graph=env.state.record(),
            streams_after=stream_table(env.state),
        )
        trace.append(entry)
    return trace
