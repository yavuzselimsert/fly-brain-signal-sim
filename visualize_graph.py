import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def load_graph():
    return nx.read_gml("circuit_graph.gml", destringizer=int)


def visualize_network(G):
    """
    Visualize the circuit as a network graph, with node size
    reflecting total synaptic degree and color reflecting
    betweenness centrality (bridge importance).
    """
    print("Layout hesaplanıyor (biraz zaman alabilir)...")

    # spring_layout: a force-directed layout - connected nodes are
    # pulled together, unconnected nodes are pushed apart. This is
    # the standard way to visualize network structure.
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Node size: total synaptic degree (in + out)
    total_degree = dict(G.degree(weight="weight"))
    sizes = [total_degree[n] * 0.05 for n in G.nodes()]

    # Node color: betweenness centrality (bridge importance)
    print("Centrality hesaplanıyor...")
    centrality = nx.betweenness_centrality(G, weight="weight")
    colors = [centrality[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 14))

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=sizes,
        node_color=colors,
        cmap=cm.viridis,
        alpha=0.85,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.08,
        arrows=True,
        arrowsize=5,
        width=0.5,
        ax=ax,
    )

           # Label only the top 5 highest-degree neurons (labeling more
    # becomes unreadable due to overlap in a dense network)
    top_nodes = sorted(total_degree.items(), key=lambda x: x[1], reverse=True)[:5]

    # Manually staggered offsets (dx, dy) so labels don't overlap
    # even when nodes are close together.
    offsets = [
        (0.15, 0.10),
        (0.15, -0.10),
        (-0.18, 0.10),
        (-0.18, -0.10),
        (0.0, 0.18),
    ]

    for (node_id, _), (dx, dy) in zip(top_nodes, offsets):
        x, y = pos[node_id]
        label = f"{G.nodes[node_id]['type']} ({node_id})"

        label_x, label_y = x + dx, y + dy

        # Leader line connecting the node to its label
        ax.plot([x, label_x], [y, label_y], color="gray", linewidth=0.7, zorder=4)

        ax.annotate(
            label,
            xy=(label_x, label_y),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.95),
        )
   

    plt.colorbar(nodes, ax=ax, label="Betweenness Centrality", shrink=0.7)
    ax.set_title(
        "Drosophila Olfactory Projection Neuron Circuit\n"
        "(node size = synaptic degree, color = bridge importance)",
        fontsize=13,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("circuit_network.png", dpi=150)
    print("\nGrafik circuit_network.png olarak kaydedildi.")


def main():
    G = load_graph()
    visualize_network(G)


if __name__ == "__main__":
    main()