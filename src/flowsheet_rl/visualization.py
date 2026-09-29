"""Headless, exportable plots; no external layout engine required."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_flowsheet(record, path):
    graph = nx.DiGraph()
    labels = {}
    for node in record["nodes"]:
        graph.add_node(node["id"])
        labels[node["id"]] = f"{node['id']}: {node['kind']}"
    for edge in record["edges"]:
        graph.add_edge(edge["source"], edge["target"])
    generations = list(nx.topological_generations(graph))
    pos = {
        n: (level, -i + (len(nodes) - 1) / 2)
        for level, nodes in enumerate(generations)
        for i, n in enumerate(nodes)
    }
    fig, ax = plt.subplots(figsize=(max(8, len(generations) * 1.7), 4))
    nx.draw_networkx(
        graph,
        pos,
        labels=labels,
        node_size=2300,
        node_color="#d6e9ed",
        font_size=8,
        arrowsize=18,
        ax=ax,
    )
    edge_labels = {}
    for edge in record["edges"]:
        f = edge["stream"]["flow"]
        edge_labels[(edge["source"], edge["target"])] = "A/B/C\n" + "/".join(f"{v:.2f}" for v in f)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, ax=ax)
    ax.set_title("Analytical flowsheet | molar flows in synthetic units")
    ax.margins(0.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(summaries, path):
    names = [s["policy"] for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(names))
    axes[0].bar(
        x,
        [s["return_mean"] for s in summaries],
        yerr=[s["return_std"] for s in summaries],
        capsize=4,
        color="#307b91",
    )
    axes[0].set_ylabel("Episode return (mean +/- sample SD)")
    axes[1].bar(x, [s["feasible_rate"] for s in summaries], color="#519c82")
    axes[1].set_ylabel("Feasible design fraction")
    axes[1].set_ylim(0, 1.1)
    for ax in axes:
        ax.set_xticks(x, names, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Short smoke comparison | no convergence claim")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_transfer(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in sorted({r["strategy"] for r in runs}):
        selected = [r for r in runs if r["strategy"] == label]
        curves = [r["training"]["learning_curve"] for r in selected]
        steps = [c["training_steps"] for c in curves[0]]
        values = np.array([[c["return_mean"] for c in curve] for curve in curves])
        rates = np.array([[c["feasible_rate"] for c in curve] for curve in curves])
        mean = values.mean(axis=0)
        line = axes[0].plot(steps, mean, marker="o", label=label)[0]
        axes[0].fill_between(
            steps,
            mean - values.std(axis=0),
            mean + values.std(axis=0),
            alpha=0.15,
            color=line.get_color(),
        )
        axes[1].plot(steps, rates.mean(axis=0), marker="o", label=label)
    axes[0].set_ylabel("Evaluation return; band = training-seed SD")
    axes[1].set_ylabel("Feasible design fraction")
    axes[1].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.set_xlabel("Target environment training steps")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Toy transfer smoke | source pretraining budget reported separately")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
