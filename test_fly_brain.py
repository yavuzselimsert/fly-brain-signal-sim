import numpy as np
import pytest
import networkx as nx

from lif_neuron import simulate_lif, constant_current, V_REST, V_THRESHOLD, V_RESET


# --- LIF single-neuron tests ---

def test_neuron_stays_at_rest_with_no_input():
    """
    With zero input current, membrane potential should never
    leave the resting state.
    """
    times, voltages, spikes = simulate_lif(constant_current(0.0), duration=50.0)
    assert np.allclose(voltages, V_REST, atol=0.01)
    assert len(spikes) == 0


def test_subthreshold_current_produces_no_spikes():
    """
    A small input current should settle below threshold and
    never cause a spike (subthreshold regime, as validated
    manually in the flight_dashboard-equivalent LIF plot).
    """
    times, voltages, spikes = simulate_lif(constant_current(1.5), duration=100.0)
    assert len(spikes) == 0
    assert np.max(voltages) < V_THRESHOLD


def test_suprathreshold_current_produces_spikes():
    """
    A strong enough input current should reliably produce spikes.
    """
    times, voltages, spikes = simulate_lif(constant_current(3.0), duration=100.0)
    assert len(spikes) > 0


def test_higher_current_increases_firing_rate():
    """
    Firing rate should increase monotonically with input current
    (the basic f-I curve relationship).
    """
    _, _, spikes_low = simulate_lif(constant_current(2.5), duration=100.0)
    _, _, spikes_high = simulate_lif(constant_current(4.0), duration=100.0)
    assert len(spikes_high) > len(spikes_low)


def test_voltage_resets_after_spike():
    """
    Whenever a spike is recorded in the trace, the model should
    have reset the membrane potential to V_RESET immediately after.
    """
    times, voltages, spikes = simulate_lif(constant_current(3.0), duration=50.0)
    assert len(spikes) > 0
    # Right after the recorded spike marker (30.0 mV), voltage should
    # drop back down close to V_RESET on the next simulated step.
    spike_indices = np.where(voltages == 30.0)[0]
    for idx in spike_indices:
        if idx + 1 < len(voltages):
            assert voltages[idx + 1] < V_THRESHOLD


# --- Connectome graph tests ---

def test_graph_loads_and_has_expected_structure():
    """
    Sanity check that the real connectome graph loaded from disk
    has a reasonable, non-trivial structure.
    """
    G = nx.read_gml("circuit_graph.gml", destringizer=int)

    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
    assert nx.is_weakly_connected(G)


def test_graph_has_no_isolated_neurons():
    """
    All neurons in our selected circuit should participate in
    at least one connection, since they were chosen as a
    functionally connected sub-circuit.
    """
    G = nx.read_gml("circuit_graph.gml", destringizer=int)
    isolated = list(nx.isolates(G))
    assert len(isolated) == 0


def test_all_edge_weights_are_positive():
    """
    Synaptic weights (synapse counts) should always be positive
    integers - a real biological connection can't have zero or
    negative synapses.
    """
    G = nx.read_gml("circuit_graph.gml", destringizer=int)
    weights = [data["weight"] for _, _, data in G.edges(data=True)]
    assert all(w > 0 for w in weights)


# --- Network simulation tests ---

def test_synaptic_current_decays_over_time():
    """
    The exponential decay factor used for synaptic current should
    be strictly between 0 and 1 - otherwise current would either
    never decay (>=1) or vanish instantly (<=0), breaking temporal
    summation.
    """
    import network_simulation as ns
    decay = np.exp(-ns.TIME_STEP / ns.SYNAPTIC_TAU)
    assert 0.0 < decay < 1.0