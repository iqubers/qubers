#!/usr/bin/env python

from qiskit import IBMQ, Aer, QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.providers.aer.noise import NoiseModel

from qiskit.visualization import plot_coupling_map, plot_gate_map, plot_error_map, plot_histogram, plot_circuit_layout

from matplotlib import pyplot as plt
from datetime import date
import numpy as np


SHOTS = 256


# Zero circuit: get zero from measurement
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

print('\nLogical quantum circuit:')
print(qc)
qc.draw(output='mpl')
plt.show()


# noise free simulator
ideal_sim = AerSimulator()

circ = transpile(qc, ideal_sim)
result = ideal_sim.run(circ, memory=True, shots=SHOTS).result()
counts = result.get_counts(circ)

plot_histogram(counts, title='Ideal simulator')
plt.show()

results = np.array(result.get_memory(circ), dtype=int)
heads_count = results.cumsum()
shot_index = np.arange(len(results)) + 1
heads_ratio = heads_count / shot_index
heads_std = [heads_ratio[:i].std() for i in range(1, len(heads_ratio)+1)]
#tails_ratio = 1 - heads_ratio


# get a real backend from a real provider
provider = IBMQ.load_account()
QPU = 'ibmq_belem'
backend_dev = provider.get_backend(QPU)
config = backend_dev.configuration()
noise_model = NoiseModel.from_backend(backend_dev)

print(backend_dev)
print(noise_model)
print('''Physical backend:
    %s %s
    n_qubits: %d
    supports OpenPulse: %s
    basis gates: %s''' %
    (config.backend_name, config.backend_version, config.n_qubits, 'yes' if config.open_pulse else 'no', config.basis_gates))

plot_gate_map(backend_dev)
plt.title('Gate map for %s' % QPU)
plt.show()
plot_error_map(backend_dev)
plt.show()


# generate a simulator that mimics the real quantum system with the latest calibration results
backend_sim = AerSimulator.from_backend(backend_dev)
#backend_sim.set_options(device='GPU')
config = backend_sim.configuration()
#props = backend_sim.properties()

print('''Simulation backend:
    %s %s
    n_qubits: %d
    supports OpenPulse: %s
    basis gates: %s''' %
    (config.backend_name, config.backend_version, config.n_qubits, 'yes' if config.open_pulse else 'no', config.basis_gates))

circ = transpile(qc, backend_sim)
#qi = QuantumInstance(backend=backend_sim, noise_model=noise_model)
#print(qi)

plot_circuit_layout(circ, backend_dev)
plt.show()

result = backend_sim.run(circ, memory=True, shots=SHOTS).result()
counts = result.get_counts(circ)

plot_histogram(counts, title='Noisy simulator (noise from %s at %s)' % (QPU, date.today()))
plt.show()

print('\nQuantum circuit on the physical gates of the QPU %s:' % QPU)
print(transpile(circ, basis_gates=config.basis_gates))
circ.decompose().draw(output='mpl')
plt.show()

skip = 20
#plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20.colors)
plt.plot(shot_index[skip:], heads_ratio[skip:], label='heads (1)')
plt.fill_between(shot_index[skip:], heads_ratio[skip:] + heads_std[skip:], heads_ratio[skip:] - heads_std[skip:], alpha=0.2)
#plt.plot(shot_index[skip:], tails_ratio[skip:], label='tails (0)')

results = np.array(result.get_memory(circ), dtype=np.byte)
heads_count = results.cumsum()
tails_count = -(results - np.byte(1)).cumsum()
shot_index = np.arange(len(results)) + 1
heads_ratio = heads_count / shot_index
heads_std = [heads_ratio[:i].std() for i in range(1, len(heads_ratio)+1)]
#tails_ratio = 1 - heads_ratio

plt.plot(shot_index[skip:], heads_ratio[skip:], label='noisy heads (1)')
plt.fill_between(shot_index[skip:], heads_ratio[skip:] + heads_std[skip:], heads_ratio[skip:] - heads_std[skip:], alpha=0.2)
#plt.plot(shot_index[skip:], tails_ratio[skip:], label='noisy tails (0)')
plt.grid(axis='y')
plt.xlabel('Shot index')
plt.ylabel('Ratio')
plt.legend()
plt.show()
