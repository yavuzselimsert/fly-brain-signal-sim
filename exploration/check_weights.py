import networkx as nx

G = nx.read_gml("circuit_graph.gml", destringizer=int)

# Find the same stimulus neuron (highest out-degree)
out_degrees = dict(G.out_degree(weight="weight"))
stimulus_node = max(out_degrees, key=out_degrees.get)

print(f"Stimulus neuron: {G.nodes[stimulus_node]['type']} (ID {stimulus_node})")

edges = list(G.out_edges(stimulus_node, data="weight"))
weights = [w for _, _, w in edges]

print(f"Number of outgoing connections: {len(edges)}")
print(f"Min weight: {min(weights)}")
print(f"Max weight: {max(weights)}")
print(f"Average weight: {sum(weights) / len(weights):.2f}")