#!/usr/bin/env python

from qiskit import IBMQ, Aer, QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel

from qiskit.visualization import plot_coupling_map, plot_gate_map, plot_error_map, plot_histogram, plot_circuit_layout

from matplotlib import pyplot as plt
from datetime import date



# Zero circuit: get zero from measurement
qc = QuantumCircuit(1, 1)
qc.measure(0, 0)

print('\nQuantum circuit diagram:')
print(qc)
qc.draw(output='mpl')
plt.show()


# no-noise simulator
ideal_sim = AerSimulator()

circ = transpile(qc, ideal_sim)
result = ideal_sim.run(circ).result()
counts = result.get_counts(circ)

plot_histogram(counts, title='Ideal simulator')
plt.show()


# get a real backend from a real provider
provider = IBMQ.load_account()
QPU = 'ibmq_belem'
backend_dev = provider.get_backend(QPU)
noise_model = NoiseModel.from_backend(backend_dev)

print(backend_dev)
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
#qi = QuantumInstance(backend=backend_sim, noise_model=noise_model)
#print(qi)

plot_circuit_layout(circ, backend_dev)
plt.show()

result = backend_sim.run(circ).result()
counts = result.get_counts(circ)

plot_histogram(counts, title='Noisy simulator (noise from %s at %s)' % (QPU, date.today()))
plt.show()

print('\nQuantum circuit on the physical dev.:')
print(circ.decompose())
circ.decompose().draw(output='mpl')
plt.show()
