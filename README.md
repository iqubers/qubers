# qubers

Scripts:

0. Hello zero:
   - *zero.py*
   - Simplest of them all, just setup a quantum circuit with 1 qubit and measure it.
  
1. Flip a coin:
   - *coin_flip_and_superposition.py*
   - How to flip a coin in the quantum realm?
   - Put a qubit in a superposition of 0 and 1 (by a Hadamard gate), and measure it. Head or tail?
    
2. The quantum weirdness of entanglement 101:
   - *two_coins_entangled.py*
   - If we can entangle two coins (qubits) so they are always in the same state no matter what (Hadamard + CNOT gates), if one is measured, the other is known. We have a so called Bell pair now.
   - That doesn't sound special or weird, right? What about if we could give one of the entangle qubits to someone at a distant place? ...
 
3. Teleportation (mic drops):
   - not  yet
   - Continuation of the previous example...
    
4. 1D atomic chain (tight-binding)
   - to be uploaded soon...
   - Fermionic operators in second quantization can be mapped to qubit operators, which means we can express Hamiltonians as quantum circuits naturally. In this simple example we set a 5-sites TB hamiltonian with 1 defect site and calculate it's time evolution from a determined initial state.

5. Hydrogen molecule dissociation profile
   - H2_PES.py
   - Setup a hydrogen molecule, and calculate it's ground state energy for various bond lengths with different methods. How does the quantum eigensolver fares in this variational problem?
