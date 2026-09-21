import pandas as pd
import networkx as nx


def load_data():
    """
    Load the neuron and connection data saved in the previous step.
    """
    neurons_df = pd.read_csv("neurons.csv")
    conn_df = pd.read_csv("connections.csv")
    return neurons_df, conn_df


def build_graph(neurons_df, conn_df):
    """
    Build a directed graph where nodes are neurons and edges are
    synaptic connections, weighted by synapse count.
    """
    G = nx.DiGraph()

    # Add nodes with metadata (neuron type, synapse counts)
    # Cast numpy types to native Python types for compatibility
    # with file formats like GML.
    for _, row in neurons_df.iterrows():
        G.add_node(
            int(row["bodyId"]),
            type=str(row["type"]),
            pre=int(row["pre"]),
            post=int(row["post"]),
        )

    aggregated = (
        conn_df.groupby(["bodyId_pre", "bodyId_post"])["weight"]
        .sum()
        .reset_index()
    )

    for _, row in aggregated.iterrows():
        source = int(row["bodyId_pre"])
        target = int(row["bodyId_post"])
        weight = int(row["weight"])

        if G.has_node(source) and G.has_node(target):
            G.add_edge(source, target, weight=weight)

    return G


def main():
    neurons_df, conn_df = load_data()
    G = build_graph(neurons_df, conn_df)

    print(f"Graf oluşturuldu.")
    print(f"Düğüm (nöron) sayısı: {G.number_of_nodes()}")
    print(f"Kenar (bağlantı) sayısı: {G.number_of_edges()}")

    # Basic sanity checks
    isolated = list(nx.isolates(G))
    print(f"\nİzole (bağlantısız) nöron sayısı: {len(isolated)}")

    # Save the graph for later steps
    nx.write_gml(G, "circuit_graph.gml")
    print("\nGraf circuit_graph.gml olarak kaydedildi.")


if __name__ == "__main__":
    main()