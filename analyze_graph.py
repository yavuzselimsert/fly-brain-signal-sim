import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def load_graph():
    return nx.read_gml("circuit_graph.gml", destringizer=int)


def analyze_degrees(G):
    """
    Analyze in-degree and out-degree distributions.
    In-degree: how many other neurons synapse onto this one.
    Out-degree: how many other neurons this one synapses onto.
    """
    in_degrees = dict(G.in_degree(weight="weight"))
    out_degrees = dict(G.out_degree(weight="weight"))

    print("=== Top 5 neurons by incoming synapse weight (most 'listened to') ===")
    top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for node_id, degree in top_in:
        neuron_type = G.nodes[node_id]["type"]
        print(f"  {neuron_type} (ID {node_id}): {degree} incoming synapses")

    print("\n=== Top 5 neurons by outgoing synapse weight (most 'talkative') ===")
    top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for node_id, degree in top_out:
        neuron_type = G.nodes[node_id]["type"]
        print(f"  {neuron_type} (ID {node_id}): {degree} outgoing synapses")

    return in_degrees, out_degrees


def analyze_centrality(G):
    """
    Betweenness centrality identifies neurons that act as
    'bridges' - lying on many shortest paths between other neurons.
    These are structurally critical even if not high-degree.
    """
    print("\nBetweenness centrality hesaplanıyor (biraz zaman alabilir)...")
    centrality = nx.betweenness_centrality(G, weight="weight")

    print("\n=== Top 5 neurons by betweenness centrality ('bridge' neurons) ===")
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for node_id, score in top_central:
        neuron_type = G.nodes[node_id]["type"]
        print(f"  {neuron_type} (ID {node_id}): centrality = {score:.4f}")

    return centrality


def analyze_structure(G):
    """
    Basic structural properties of the network.
    """
    print("\n=== Network Structure ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")

    # Is the graph (weakly) connected as a whole?
    is_connected = nx.is_weakly_connected(G)
    print(f"Weakly connected: {is_connected}")

    if not is_connected:
        components = list(nx.weakly_connected_components(G))
        print(f"Number of connected components: {len(components)}")
        largest = max(components, key=len)
        print(f"Largest component size: {len(largest)} neurons")


def plot_degree_distribution(in_degrees, out_degrees):
    """
    Plot the distribution of in- and out-degrees.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(list(in_degrees.values()), bins=30, color="tab:blue", alpha=0.7)
    axes[0].set_xlabel("Incoming synapse weight")
    axes[0].set_ylabel("Number of neurons")
    axes[0].set_title("In-Degree Distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(list(out_degrees.values()), bins=30, color="tab:orange", alpha=0.7)
    axes[1].set_xlabel("Outgoing synapse weight")
    axes[1].set_ylabel("Number of neurons")
    axes[1].set_title("Out-Degree Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("degree_distribution.png", dpi=150)
    print("\nGrafik degree_distribution.png olarak kaydedildi.")


def main():
    G = load_graph()

    in_degrees, out_degrees = analyze_degrees(G)
    analyze_centrality(G)
    analyze_structure(G)
    plot_degree_distribution(in_degrees, out_degrees)


if __name__ == "__main__":
    main()