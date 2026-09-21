import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from network_simulation import simulate_network, load_graph
import network_simulation as ns


def run_sensitivity_analysis():
    G = load_graph()

    out_degrees = dict(G.out_degree(weight="weight"))
    stimulus_node = max(out_degrees, key=out_degrees.get)

    gains = [0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.10]

    total_spikes_list = []
    active_neurons_list = []

    for gain in gains:
        ns.SYNAPTIC_GAIN = gain  # override module-level parameter

        nodes, spike_times = simulate_network(G, stimulus_node, duration=100.0)

        total_spikes = sum(len(t) for t in spike_times.values())
        active_neurons = sum(1 for t in spike_times.values() if len(t) > 0)

        total_spikes_list.append(total_spikes)
        active_neurons_list.append(active_neurons)

        print(f"GAIN={gain:.3f} | Total spikes: {total_spikes:6d} | Active neurons: {active_neurons:3d} / {len(nodes)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(gains, total_spikes_list, "o-", color="tab:red")
    axes[0].set_xlabel("Synaptic gain")
    axes[0].set_ylabel("Total spikes (100 ms)")
    axes[0].set_title("Total Network Activity vs Synaptic Gain")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(1000, color="gray", linestyle="--", alpha=0.5, label="Runaway threshold (illustrative)")
    axes[0].legend()

    axes[1].plot(gains, active_neurons_list, "o-", color="tab:blue")
    axes[1].set_xlabel("Synaptic gain")
    axes[1].set_ylabel("Neurons active at least once")
    axes[1].set_title("Circuit Recruitment vs Synaptic Gain")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gain_sensitivity.png", dpi=150)
    print("\nPlot saved as gain_sensitivity.png")


if __name__ == "__main__":
    run_sensitivity_analysis()