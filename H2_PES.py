#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# qiskit
from qiskit import Aer, IBMQ
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.providers.aer import AerSimulator
#from qiskit.algorithms.minimum_eigen_solvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import ExcitationPreserving
from qiskit.algorithms import NumPyMinimumEigensolver, VQE

# qiskit nature imports
from qiskit_nature.settings import settings
settings.dict_aux_operators = True
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber, ElectronicEnergy
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer, ActiveSpaceTransformer
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.algorithms import GroundStateEigensolver, VQEUCCFactory
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.algorithms.pes_samplers import BOPESSampler, Extrapolator
from qiskit.algorithms.optimizers import SPSA, P_BFGS, COBYLA, ADAM, SLSQP, L_BFGS_B, QNSPSA
from qiskit_nature.circuit.library import UCCSD, HartreeFock

from scipy import interpolate
from scipy.optimize import minimize
from pyscf import gto, scf, fci, dft

import time


Ha_to_eV = 27.2114

# https://pyscf.org/_modules/pyscf/gto/basis.html
basis = 'dz'
#basis = 'sto-3g'


# Classical algorithms: Hartree-Fock, DFT and Full Configuration Interaction
print('\nRunning classical algorithms:')
dist = np.concatenate([np.linspace(.4, 1.5, 7), np.linspace(2., 4.5, 6)])
E_HF, E_KS, E_FCI = [], [], []
for d in dist:
    mol = gto.M(
        atom = [['H', [.0, .0, .0]], ['H', [.0, .0, d]]],
        basis = basis,
        symmetry = True,
        charge=0,
        spin=0
    )

    myhf = scf.HF(mol).run()
    E_HF.append(myhf.e_tot)
    mydft = dft.KS(mol, xc='b3lyp').run()
    E_KS.append(mydft.e_tot)
    myci = fci.FCI(myhf).run()
    E_FCI.append(myci.e_tot)

E_HF = np.array(E_HF) * Ha_to_eV
E_KS = np.array(E_KS) * Ha_to_eV
E_FCI = np.array(E_FCI) * Ha_to_eV

print('\nBasis set:\n', '\n '.join(mol.ao_labels()))
print(mol.basis)


# Quantum algorithm
print('\n\nRunning quantum algorithms:')

d0 = .4
points = np.concatenate([np.linspace(0., 1, 30), np.linspace(1.1, 4.1, 21)])
distances = points + d0
#print(distances)

stretch = partial(Molecule.absolute_stretching, atom_pair=(1, 0))
mol = Molecule(
    geometry=[['H', [.0, .0, .0]], ['H', [.0, .0, d0]]],
    degrees_of_freedom=[stretch],
    charge=0,
    multiplicity=1
)

driver = ElectronicStructureMoleculeDriver(
    mol, basis=basis,
    driver_type=ElectronicStructureDriverType.PYSCF)

properties = driver.run()
print('\n', properties)

transformers = []
#transformers += [FreezeCoreTransformer(remove_orbitals=[])]
#transformers += [ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=3)]
es_problem = ElectronicStructureProblem(driver, transformers=transformers)

second_q_ops = es_problem.second_q_ops()
#for prop, q_op in second_q_ops.items():
    #print(prop, q_op)

FermionicOp.set_truncation(0)
print('\nElectronic Energy', second_q_ops['ElectronicEnergy'])

num_particles = es_problem.num_particles
num_spin_orbitals = es_problem.num_spin_orbitals
print('\nNumber of particles:', num_particles)
print('Number of spin orbitals:', num_spin_orbitals)

mapper = ParityMapper() # JordanWignerMapper()
qubit_converter = QubitConverter(mapper, two_qubit_reduction=True)

## noise free simulator
backend = Aer.get_backend("aer_simulator_statevector")
basis_gates = None
error_mitigation = None

# noisy simulator
#provider = IBMQ.load_account()
#backend_dev = provider.get_backend('ibmq_belem')
#basis_gates = backend_dev.configuration().basis_gates
#backend = AerSimulator.from_backend(backend_dev)
#error_mitigation = CompleteMeasFitter

qi = QuantumInstance(backend=backend,
                     measurement_error_mitigation_cls=error_mitigation,
                     basis_gates=basis_gates)

#solver = VQE(quantum_instance=qi)
opt = COBYLA(maxiter=3000)
#init_state = HartreeFock(num_spin_orbitals, num_particles, qubit_converter)
#ansatz = UCCSD(qubit_converter, num_particles, num_spin_orbitals, initial_state=init_state)
##solver = VQEUCCFactory(quantum_instance=qi, optimizer=opt, include_custom=False, ansatz=ansatz, initial_state=init_state)
solver = VQEUCCFactory(quantum_instance=qi, optimizer=opt)

# set logging if you want it
#import logging
#logging.basicConfig(level=logging.INFO)
#logging.getLogger('qiskit.algorithms.minimum_eigen_solvers.vqe').setLevel(logging.INFO)


gs = GroundStateEigensolver(qubit_converter, solver)

# define extrapolators
extrap_poly = Extrapolator.factory('poly', degree=1)
extrap = Extrapolator.factory('window', extrapolator=extrap_poly)

bs = BOPESSampler(gs, bootstrap=True, num_bootstrap=None, extrapolator=extrap)
t0 = time.time()
res = bs.sample(es_problem, points)
print('PES sampled in %.2f s' % (time.time() - t0))

## exact numpy solver
#solver_numpy = NumPyMinimumEigensolver()
#gs_numpy = GroundStateEigensolver(qubit_converter, solver_numpy)
#bs_classical = BOPESSampler(gs_numpy, bootstrap=False, num_bootstrap=None, extrapolator=None)
#res_np = bs_classical.sample(es_problem, points)
#print(res_np.energies)
#E_np = np.array(res_np.energies) * Ha_to_eV

#f = interpolate.interp1d(distances, np.array(res_np.energies, dtype=float), kind='cubic')
#fmin = minimize(f, [1.])
#print('Minimum at:', fmin.x, f(fmin.x))


fig = plt.figure()
E = np.array(res.energies) * Ha_to_eV

for y, l, m in zip([E_HF, E_KS, E_FCI], ['HF', 'DFT(B3LYP)', 'Full CI'], ['x', 's', 'o']):
    plt.scatter(dist, y - y[-1], marker=m, label=l, alpha=0.3)
plt.plot(distances, E - E[-1], label='VQE') # , linestyle='dashed'
plt.title('Dissociation profile')
plt.xlabel('H-H bond length ($\AA$)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()
