#!/usr/bin/env python

from qiskit import IBMQ, Aer, QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.providers.aer.noise import NoiseModel
#from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit_experiments.library import LocalReadoutError

from qiskit.visualization import plot_coupling_map, plot_gate_map, plot_error_map, plot_histogram, plot_circuit_layout

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from datetime import date
import numpy as np


SHOTS = 200


def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()



# Two qubits circuit: entangle two coins
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
#qc.measure([0,1], [0,1])

qr = qc.qubits
n_qubits = len(qr)
active_qubits_list = list(range(n_qubits))

print('\nLogical quantum circuit:')
print(qc)
#qc.draw(output='mpl')
#plt.show()



# Noise free simulator
ideal_sim = AerSimulator()

circ = transpile(qc, ideal_sim)
ideal_result = ideal_sim.run(circ, memory=True, shots=200).result()
ideal_counts = ideal_result.get_counts(circ)
ideal_probs = {label: count/200 for label, count in ideal_counts.items()}

ideal_results = np.array(ideal_result.get_memory(circ))
ideal_states = np.unique(ideal_results)
shot_index = np.arange(len(ideal_results)) + 1
ideal_states_count, ideal_states_ratio, ideal_states_std = {}, {}, {}
for state in ideal_states:
    ideal_states_count[state] = np.zeros((len(ideal_results)), dtype=int)
    ideal_states_count[state][ideal_results==state] = 1
    ideal_states_count[state] = ideal_states_count[state].cumsum()
    ideal_states_ratio[state] = ideal_states_count[state] / shot_index
    ideal_states_std[state] = [ideal_states_ratio[state][:i].std() for i in range(1, len(ideal_results)+1)]


# get a real backend from a real provider
provider = IBMQ.load_account()
QPU = 'ibmq_belem'

backend_dev = provider.get_backend(QPU)
config = backend_dev.configuration()
#props = backend_dev.properties()
noise_model = NoiseModel.from_backend(backend_dev)

print('''\nBackend:
    %s %s
    n_qubits: %d
    supports OpenPulse: %s
    basis gates: %s\n''' %
    (config.backend_name, config.backend_version, config.n_qubits, 'yes' if config.open_pulse else 'no', config.basis_gates))
print(noise_model)

plot_gate_map(backend_dev)
plt.title('Gate map for %s' % QPU)
plt.show()

plot_error_map(backend_dev)
plt.show()


# generate a simulator that mimics the real quantum system with the latest calibration results
backend_sim = AerSimulator.from_backend(backend_dev)
#backend_sim.set_options(device='GPU')

circ = transpile(qc, backend_sim)

plot_circuit_layout(circ, backend_dev)
plt.show()

result = backend_sim.run(circ, memory=True, shots=SHOTS, method='density_matrix').result()
counts = result.get_counts(circ)
probs = {label: count/SHOTS for label, count in counts.items()}

plot_histogram([ideal_counts, counts], legend=['ideal', 'noisy'],
               title='Ideal vs. Noisy sim. (from %s at %s)' % (QPU, date.today()))
plt.show()

print('\nTranspiled circuit')
print(circ.decompose())
print('\nCircuit on the actual physical gates of the QPU %s:' % QPU)
print(transpile(circ, basis_gates=config.basis_gates))
#circ.decompose().draw(output='mpl')
#plt.show()


# Error mitigation
qubits_list = list(range(config.n_qubits))

exp = LocalReadoutError(qubits_list)
print('\nLocal error readout mitigation experiment\'s circuits:')
for c in exp.circuits():
    print(c)

exp.analysis.set_options(plot=True)
exp_result = exp.run(backend_sim)
mitigator = exp_result.analysis_results(0).value
print(exp_result)

show_figure(exp_result.figure(0).figure)

mitigated_quasi_probs = mitigator.quasi_probabilities(counts, qubits=active_qubits_list, clbits=active_qubits_list)
mitigated_stddev = mitigated_quasi_probs._stddev_upper_bound
mitigated_probs = (mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities())

plot_histogram([ideal_probs, probs, mitigated_probs], legend=['ideal', 'noisy', 'mitigated'], title='Ideal, noisy, mitigated')
plt.tight_layout()
plt.show()
