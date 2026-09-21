import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm


def load_results():
    with open("simulation_results.pkl", "rb") as f:
        data = pickle.load(f)
    return data["nodes"], data["spike_times"], data["stimulus_node"]


def load_graph():
    return nx.read_gml("circuit_graph.gml", destringizer=int)


def plot_raster(nodes, spike_times, stimulus_node, G):
    """
    Raster plot: one row per neuron that fired, one dot per spike.
    Neurons are ordered by their first spike time, so the plot
    visually reads as a top-to-bottom propagation wave.
    """
    active = {n: t for n, t in spike_times.items() if len(t) > 0}
    ordered = sorted(active.items(), key=lambda x: x[1][0])

    fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.3)))

    labels = []
    for i, (node_id, times) in enumerate(ordered):
        color = "red" if node_id == stimulus_node else "tab:blue"
        ax.scatter(times, [i] * len(times), color=color, s=40, marker="|")

        neuron_type = G.nodes[node_id]["type"]
        tag = f"{neuron_type} (stimulus)" if node_id == stimulus_node else neuron_type
        labels.append(tag)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Spike Raster Plot (signal propagation through the circuit)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("raster_plot.png", dpi=150)
    print("Raster plot saved as raster_plot.png")


def create_network_animation(nodes, spike_times, stimulus_node, G):
    """
    Animate the spread of activity across the circuit graph.
    Neurons light up briefly when they spike.
    """
    print("Computing layout...")
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Build a flat, time-ordered list of (time, node) spike events
    events = []
    for node_id, times in spike_times.items():
        for t in times:
            events.append((t, node_id))
    events.sort()

    if not events:
        print("No spikes to animate.")
        return

    max_time = max(t for t, _ in events)
    frame_dt = 1.0  # ms per animation frame
    n_frames = int(max_time / frame_dt) + 10

    # How long a neuron stays "lit up" after spiking (visual persistence)
    glow_duration = 5.0  # ms

    fig, ax = plt.subplots(figsize=(10, 10))

    base_color = "lightgray"
    stimulus_color = "red"
    active_color = "orange"

    def get_node_colors(current_time):
        colors = []
        for n in G.nodes():
            if n == stimulus_node:
                colors.append(stimulus_color)
                continue

            recently_fired = any(
                0 <= current_time - t < glow_duration
                for t in spike_times.get(n, [])
            )
            colors.append(active_color if recently_fired else base_color)
        return colors

    def get_node_sizes(current_time):
        sizes = []
        for n in G.nodes():
            recently_fired = any(
                0 <= current_time - t < glow_duration
                for t in spike_times.get(n, [])
            )
            base = 80 if n == stimulus_node else 20
            sizes.append(base * 2.5 if recently_fired else base)
        return sizes

    nx.draw_networkx_edges(G, pos, alpha=0.05, arrows=False, width=0.5, ax=ax)
    node_artist = nx.draw_networkx_nodes(
        G, pos, node_color=base_color, node_size=20, ax=ax
    )
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, va="top")

    ax.set_title("Signal Propagation Through Olfactory Projection Neuron Circuit")
    ax.axis("off")

    def update(frame):
        current_time = frame * frame_dt
        node_artist.set_color(get_node_colors(current_time))
        node_artist.set_sizes(get_node_sizes(current_time))
        time_text.set_text(f"t = {current_time:.1f} ms")
        return node_artist, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=50, blit=False
    )

    anim.save("signal_propagation.gif", writer="pillow", fps=15)
    print("Animation saved as signal_propagation.gif")


def main():
    nodes, spike_times, stimulus_node = load_results()
    G = load_graph()

    plot_raster(nodes, spike_times, stimulus_node, G)
    create_network_animation(nodes, spike_times, stimulus_node, G)


if __name__ == "__main__":
    main()