
# Import from Qiskit
from qiskit_aer import AerSimulator
#from qiskit.utils import QuantumInstance
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal, ExcitationPreserving
# Import from Qiskit Aqua
from scipy.optimize import fmin_l_bfgs_b

# Import from Qiskit Chemistry
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA 
# Import from other python libraries
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
# Simulators
SV = "statevector"
QA = "qasm"

# Mappings
PA = "parity"
JW = "jordan_wigner"
BK = "bravyi_kitaev"

#Optimizers
CO = "COBYLA"
BF = "L_BFGS_B"
SL = "SLSQP"
SP = "SPSA"
"""
# Quantum Instances for running the quantum circuit
quantum_instances = {
    SV: QuantumInstance(backend=BasicAer.get_backend('statevector_simulator')),
    QA: QuantumInstance(backend=Aer.get_backend('qasm_simulator'))}
"""
# Mappings for transforming the molecular Hamiltonian into a set of Pauli products acting on qubits
map_types = {
    PA: "parity",
    JW: "jordan_wigner",
    BK: "bravyi_kitaev"}
    
# Classical Optimizers for parameter minimization
MAX_ITER = 500
optimizers = {
    CO: COBYLA(maxiter=MAX_ITER),
    BF: fmin_l_bfgs_b(maxiter=MAX_ITER),
    SL: SLSQP(maxiter=MAX_ITER),
    SP: SPSA(max_trials=MAX_ITER,save_steps=100)}


def get_qubit_op_H2(dist, basis, map_type, verbose=False,tqr=False):
    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, charge=0, spin=0, basis=basis)
    molecule = driver.run()
    #Calculate nuclear repulsion energy
    repulsion_energy = molecule.nuclear_repulsion_energy 
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    #Calculate one and two body integrals
    h1=molecule.one_body_integrals
    h2=molecule.two_body_integrals
    ferOp = FermionicOperator(h1=h1, h2=h2) 
    #Perform the Mapping from Fermionic operators to Qubit operators
    qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
    shift =  repulsion_energy
    if verbose:
        print(h1)
        print(h2)
        print(qubitOp)
        print(qubitOp.print_details())
    if tqr:
        qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
    if verbose:
        print(qubitOp)
        print(qubitOp.print_details())
    return qubitOp, num_particles, num_spin_orbitals, shift


def build_ansatz(qubitOp, num_orbitals, num_particles, map_type="parity", initial_state="HartreeFock", var_form="UCCSD", depth=1,tqr=False):
    # Specify your initial state
    initial_states = {
        'Zero': Zero(qubitOp.num_qubits),
        'HartreeFock': HartreeFock(num_orbitals, num_particles, map_type)}
    # Select a state preparation ansatz
    # Equivalently, choose a parameterization for the trial wave function.
    var_forms = {
        'UCCSD': UCCSD(num_orbitals=num_orbitals, num_particles=num_particles,initial_state=initial_states[initial_state], qubit_mapping=map_type, reps=depth,two_qubit_reduction=tqr),
        'RealAmplitudes': RealAmplitudes(qubitOp.num_qubits, reps=depth,initial_state=initial_states[initial_state]), 
        'EfficientSU2': EfficientSU2(qubitOp.num_qubits, reps=depth,initial_state=initial_states[initial_state]), 
        'TwoLocal': TwoLocal(qubitOp.num_qubits,['ry','rz'], 'cz', 'full', reps=depth,initial_state=initial_states[initial_state]), 
        'ExcitationPreserving': ExcitationPreserving(qubitOp.num_qubits,mode='iswap', entanglement='full',reps=depth,initial_state=initial_states[initial_state])}
    return var_forms[var_form]


def exact_solver(qubitOp, verbose=False):
    ee = NumPyEigensolver(qubitOp)
    result = ee.run()
    ref = result['eigenvalues'].real[0]
    if verbose:
        print('Reference value: {}'.format(ref))
    return ref


def random_initial_point(ansatz, interval=[0,1]):
    initial_point = []
    if len(interval)>1:
        for i in range(ansatz._num_parameters): # For UCCSD
            initial_point.append(random(interval[0], interval[1]))
    else:
        for i in range(ansatz._num_parameters):
            initial_point.append(random(0, interval[0]))
    return initial_point


