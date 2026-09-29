"""Directed process topology with explicit open boundaries; no node ID as a feature."""

from copy import deepcopy
from dataclasses import dataclass

import networkx as nx
import numpy as np
from gymnasium.spaces import GraphInstance

from .chemistry import Stream

NODE_TYPES = (
    "feed",
    "open",
    "mixer",
    "thermal",
    "reactor",
    "separator",
    "product",
    "purge",
    "junction",
)
EQUIPMENT = ("mixer", "thermal", "reactor", "separator")
NODE_DIM = len(NODE_TYPES) + 6
EDGE_DIM = 9
GLOBAL_DIM = 9


@dataclass
class FlowsheetGraph:
    graph: nx.DiGraph
    step: int = 0

    @classmethod
    def initial(cls, feed_a, feed_b, temperature=300.0):
        graph = nx.DiGraph()
        for i, flow in enumerate(([feed_a, 0, 0], [0, feed_b, 0])):
            source, boundary = 2 * i, 2 * i + 1
            stream = Stream(np.array(flow), temperature)
            graph.add_node(source, kind="feed", feed=stream, stream=stream, parameter=0.0)
            graph.add_node(boundary, kind="open", stream=stream, parameter=0.0)
            graph.add_edge(source, boundary, port=0, stream=stream)
        return cls(graph)

    @classmethod
    def from_record(cls, record):
        graph = nx.DiGraph()
        for node in record["nodes"]:
            attrs = {
                k: Stream(**v) if k in ("stream", "feed") else v
                for k, v in node.items()
                if k != "id"
            }
            graph.add_node(node["id"], **attrs)
        for edge in record["edges"]:
            graph.add_edge(
                edge["source"], edge["target"], port=edge["port"], stream=Stream(**edge["stream"])
            )
        state = cls(graph, record["step"])
        state.validate()
        return state

    @property
    def open_nodes(self):
        return sorted(n for n, a in self.graph.nodes(data=True) if a["kind"] == "open")

    @property
    def num_units(self):
        return sum(a["kind"] in EQUIPMENT for _, a in self.graph.nodes(data=True))

    @property
    def feed_flow(self):
        return sum(
            (a["feed"].flow for _, a in self.graph.nodes(data=True) if a["kind"] == "feed"),
            np.zeros(3),
        )

    def copy(self):
        return deepcopy(self)

    def add_unit(self, selected, kind, parameter=0.0, partner=None):
        if selected not in self.open_nodes or kind not in EQUIPMENT + ("product", "purge"):
            raise ValueError("invalid_location_or_unit")
        if kind == "mixer" and (partner not in self.open_nodes or partner == selected):
            raise ValueError("invalid_mixer_partner")
        self.graph.nodes[selected].update(kind=kind, parameter=float(parameter))
        if kind == "mixer":
            # Keep the second boundary as a zero-cost connector, preserving stable IDs.
            self.graph.nodes[partner]["kind"] = "junction"
            self.graph.add_edge(partner, selected, port=0)
        for port in range(2 if kind == "separator" else int(kind in EQUIPMENT)):
            new_id = len(self.graph)
            self.graph.add_node(new_id, kind="open", parameter=0.0)
            self.graph.add_edge(selected, new_id, port=port)
        self.validate()

    def validate(self):
        g = self.graph
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("cycle_not_supported")
        for n, a in g.nodes(data=True):
            kind = a["kind"]
            if kind not in NODE_TYPES:
                raise ValueError("unknown_node_type")
            expected_in = 0 if kind == "feed" else 2 if kind == "mixer" else 1
            expected_out = (
                0 if kind in ("open", "product", "purge") else 2 if kind == "separator" else 1
            )
            if g.in_degree(n) != expected_in or g.out_degree(n) != expected_out:
                raise ValueError("port_cardinality")
            ports = sorted(g.edges[n, dst]["port"] for dst in g.successors(n))
            if ports != list(range(expected_out)):
                raise ValueError("duplicate_or_missing_output_port")

    def graph_instance(self):
        ids = sorted(self.graph)
        lookup = {n: i for i, n in enumerate(ids)}
        x = []
        for n in ids:
            a = self.graph.nodes[n]
            s = a["stream"]
            one_hot = [float(a["kind"] == kind) for kind in NODE_TYPES]
            x.append(
                one_hot
                + [
                    float(a["kind"] == "open"),
                    (s.temperature - 280) / 170,
                    s.pressure,
                    a.get("parameter", 0.0),
                    a.get("performance", 0.0),
                    1.0,
                ]
            )
        edges, links = [], []
        for source, target, a in sorted(self.graph.edges(data=True)):
            s = a["stream"]
            edges.append(
                [
                    s.total / 2,
                    *s.composition.tolist(),
                    (s.temperature - 280) / 170,
                    s.pressure,
                    float(self.graph.nodes[target]["kind"] == "open"),
                    0.0,
                    1.0,
                ]
            )
            links.append([lookup[source], lookup[target]])
        return GraphInstance(
            np.asarray(x, dtype=np.float32),
            np.asarray(edges, dtype=np.float32).reshape(-1, EDGE_DIM),
            np.asarray(links, dtype=np.int64).reshape(-1, 2),
        )

    def record(self):
        # Stable ID-based serialization for replay, not a graph-isomorphism canonicalizer.
        nodes = []
        for n, a in sorted(self.graph.nodes(data=True)):
            nodes.append(
                {"id": n, **{k: v.record() if isinstance(v, Stream) else v for k, v in a.items()}}
            )
        edges = [
            {"source": s, "target": t, "port": a["port"], "stream": a["stream"].record()}
            for s, t, a in sorted(self.graph.edges(data=True))
        ]
        return {"step": self.step, "nodes": nodes, "edges": edges}


def to_pyg(observation):
    import torch
    from torch_geometric.data import Data

    graph = observation["graph"]
    return Data(
        x=torch.as_tensor(graph.nodes, dtype=torch.float32),
        edge_index=torch.as_tensor(graph.edge_links.T.copy(), dtype=torch.long),
        edge_attr=torch.as_tensor(graph.edges, dtype=torch.float32),
        globals=torch.as_tensor(observation["globals"], dtype=torch.float32).view(1, -1),
    )
