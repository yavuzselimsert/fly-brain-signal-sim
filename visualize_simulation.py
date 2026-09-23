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
    Neurons light up when they spike, with a visible glow period,
    and their outgoing edges highlight briefly too, making the
    propagation path easier to follow.
    """
    print("Computing layout...")
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    events = []
    for node_id, times in spike_times.items():
        for t in times:
            events.append((t, node_id))
    events.sort()

    if not events:
        print("No spikes to animate.")
        return

    max_time = max(t for t, _ in events)
    frame_dt = 1.0
    n_frames = int(max_time / frame_dt) + 15

    glow_duration = 8.0  # ms - how long a neuron stays highlighted

    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    base_color = "#2a2a2a"
    stimulus_color = "#ff3333"
    active_color = "#ffcc00"

    # Draw all edges once, very faint
    nx.draw_networkx_edges(
        G, pos, alpha=0.03, arrows=False, width=0.4,
        edge_color="white", ax=ax
    )

    node_artist = nx.draw_networkx_nodes(
        G, pos, node_color=base_color, node_size=25,
        edgecolors="none", ax=ax
    )

    time_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=14,
        color="white", va="top", fontweight="bold"
    )
    count_text = ax.text(
        0.02, 0.92, "", transform=ax.transAxes, fontsize=11,
        color="#aaaaaa", va="top"
    )

    ax.set_title(
        "Signal Propagation Through Olfactory Projection Neuron Circuit",
        color="white", fontsize=13, pad=15
    )
    ax.axis("off")

    def get_recently_fired(current_time):
        fired = set()
        for n, times in spike_times.items():
            if any(0 <= current_time - t < glow_duration for t in times):
                fired.add(n)
        return fired

    def update(frame):
        current_time = frame * frame_dt
        recently_fired = get_recently_fired(current_time)

        colors = []
        sizes = []
        for n in G.nodes():
            if n == stimulus_node:
                colors.append(stimulus_color)
                sizes.append(220 if n in recently_fired else 130)
            elif n in recently_fired:
                colors.append(active_color)
                sizes.append(180)
            else:
                colors.append(base_color)
                sizes.append(25)

        node_artist.set_color(colors)
        node_artist.set_sizes(sizes)

        time_text.set_text(f"t = {current_time:.1f} ms")
        count_text.set_text(f"Active neurons: {len(recently_fired)}")

        return node_artist, time_text, count_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=60, blit=False
    )

    anim.save(
        "signal_propagation.gif", writer="pillow", fps=15,
        savefig_kwargs={"facecolor": "black"}
    )
    print("Animation saved as signal_propagation.gif")


def main():
    nodes, spike_times, stimulus_node = load_results()
    G = load_graph()

    plot_raster(nodes, spike_times, stimulus_node, G)
    create_network_animation(nodes, spike_times, stimulus_node, G)


if __name__ == "__main__":
    main()