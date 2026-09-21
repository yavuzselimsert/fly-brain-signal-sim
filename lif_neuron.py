import numpy as np
import matplotlib.pyplot as plt


# LIF model parameters (typical values from neuroscience literature)
TAU = 10.0          # membrane time constant (ms)
V_REST = -70.0       # resting potential (mV)
V_THRESHOLD = -50.0  # spike threshold (mV)
V_RESET = -75.0      # reset potential after a spike (mV)
R = 10.0             # membrane resistance (MOhm)

TIME_STEP = 0.1      # ms


def simulate_lif(input_current_func, duration=100.0):
    """
    Simulate a single LIF neuron driven by a given input current
    function I(t), using Euler integration.

    Returns time, voltage trace, and list of spike times.
    """
    steps = int(duration / TIME_STEP)

    t = 0.0
    v = V_REST

    times = np.zeros(steps)
    voltages = np.zeros(steps)
    spike_times = []

    for i in range(steps):
        times[i] = t
        voltages[i] = v

        I = input_current_func(t)

        dv = (-(v - V_REST) + R * I) / TAU
        v += dv * TIME_STEP

        if v >= V_THRESHOLD:
            v = V_RESET
            spike_times.append(t)
            voltages[i] = 30.0  # draw a visible spike spike in the plot

        t += TIME_STEP

    return times, voltages, spike_times


def constant_current(amplitude):
    """
    Returns a function representing a constant input current.
    """
    def I(t):
        return amplitude
    return I


def main():
    # Test with a few different constant current amplitudes
    amplitudes = [1.5, 2.0, 3.0]

    fig, axes = plt.subplots(len(amplitudes), 1, figsize=(10, 8), sharex=True)

    for ax, amp in zip(axes, amplitudes):
        times, voltages, spikes = simulate_lif(constant_current(amp), duration=100.0)

        firing_rate = len(spikes) / (100.0 / 1000.0)  # spikes per second (Hz)

        ax.plot(times, voltages, color="tab:blue", linewidth=1)
        ax.axhline(V_THRESHOLD, color="red", linestyle="--", alpha=0.5, label="Threshold")
        ax.set_ylabel("V (mV)")
        ax.set_title(f"Input current = {amp} nA  |  Firing rate = {firing_rate:.1f} Hz")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        print(f"Current={amp} nA: {len(spikes)} spikes, firing rate = {firing_rate:.1f} Hz")

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig("lif_single_neuron.png", dpi=150)
    print("\nPlot saved as lif_single_neuron.png")


if __name__ == "__main__":
    main()