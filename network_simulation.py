import numpy as np
import networkx as nx
import pickle


# LIF parameters (same as single-neuron model)
TAU = 10.0
V_REST = -70.0
V_THRESHOLD = -50.0
V_RESET = -75.0
R = 10.0

TIME_STEP = 0.1

# Network-specific parameters
SYNAPTIC_GAIN = 0.04    # nA of current per unit of synaptic weight
SYNAPTIC_TAU = 5.0    # ms, how long synaptic current persists after a spike
STIMULUS_CURRENT = 6.0   # nA, injected into the stimulus neuron
STIMULUS_DURATION = 30.0  # ms


def load_graph():
    return nx.read_gml("circuit_graph.gml", destringizer=int)


def simulate_network(G, stimulus_node, duration=100.0):
    """
    Simulate the entire circuit as a network of coupled LIF neurons.
    Synaptic input is modeled as a current that decays exponentially
    over a synaptic time constant, allowing temporal summation of
    successive spikes (rather than instantaneous, single-step pulses).
    """
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    n_neurons = len(nodes)

    out_edges = {n: [] for n in nodes}
    for u, v, data in G.edges(data=True):
        out_edges[u].append((v, data["weight"]))

    steps = int(duration / TIME_STEP)

    V = np.full(n_neurons, V_REST)
    syn_current = np.zeros(n_neurons)  # decaying synaptic current per neuron

    spike_times = {n: [] for n in nodes}
    stimulus_idx = node_index[stimulus_node]

    syn_decay = np.exp(-TIME_STEP / SYNAPTIC_TAU)

    for step in range(steps):
        t = step * TIME_STEP

        # Synaptic current decays exponentially each step
        syn_current *= syn_decay

        I = syn_current.copy()

        if t < STIMULUS_DURATION:
            I[stimulus_idx] += STIMULUS_CURRENT

        dV = (-(V - V_REST) + R * I) / TAU
        V += dV * TIME_STEP

        fired_indices = np.where(V >= V_THRESHOLD)[0]

        for idx in fired_indices:
            node = nodes[idx]
            spike_times[node].append(t)
            V[idx] = V_RESET

            for target, weight in out_edges[node]:
                syn_current[node_index[target]] += SYNAPTIC_GAIN * weight

    return nodes, spike_times

def main():
    G = load_graph()

    # Choose a stimulus neuron: one of the highest out-degree
    # ("talkative") neurons found in the earlier analysis.
    out_degrees = dict(G.out_degree(weight="weight"))
    stimulus_node = max(out_degrees, key=out_degrees.get)
    stimulus_type = G.nodes[stimulus_node]["type"]

    print(f"Stimulus neuron: {stimulus_type} (ID {stimulus_node})")
    print("Simulating network...\n")

    nodes, spike_times = simulate_network(G, stimulus_node, duration=100.0)

    total_spikes = sum(len(times) for times in spike_times.values())
    active_neurons = sum(1 for times in spike_times.values() if len(times) > 0)

    print(f"Total spikes across the network: {total_spikes}")
    print(f"Neurons that fired at least once: {active_neurons} / {len(nodes)}")

    # Show the first 10 neurons to fire, in order (propagation order)
    first_spike = {
        n: times[0] for n, times in spike_times.items() if len(times) > 0
    }
    ordered = sorted(first_spike.items(), key=lambda x: x[1])[:10]

    print("\nFirst 10 neurons to fire (propagation order):")
    for node_id, spike_t in ordered:
        neuron_type = G.nodes[node_id]["type"]
        print(f"  t={spike_t:6.1f} ms  |  {neuron_type} (ID {node_id})")

    # Save results for visualization in the next step
    with open("simulation_results.pkl", "wb") as f:
        pickle.dump({
            "nodes": nodes,
            "spike_times": spike_times,
            "stimulus_node": stimulus_node,
        }, f)

    print("\nSonuçlar simulation_results.pkl olarak kaydedildi.")


if __name__ == "__main__":
    main()