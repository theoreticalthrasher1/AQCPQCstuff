from aqc_pqc import AQC_PQC
from hamiltonian import Hamiltonian 
from quantum_circuit import QCir
import networkx as nx
import numpy as np
from brute_force import Brute_Force
from qaoa_circuit import QAOA_Circuit
from aqc_qaoa import AQC_PQC_QAOA
from Quantum_Chemistry import Moleculeclass,Solvebynumpy
from aavqe import *
from qiskit_algorithms.utils import algorithm_globals
#seeds=[20, 21, 30, 33, 36, 42, 43, 55, 67,170 ]
seeds=67
algorithm_globals.random_seed= seeds
seed_transpiler = seeds
iterations = 125
shot = 6000

seed = 3
number_of_qubits = 3
steps = 50 #Choose number of steps to interpolate from initial to final Hamiltonian
connectivity = 'nearest-neighbors' #This is the connectivity of the non-parameterized gates in the Hardware Efficient ansatz
single_qubit_gates = 'ry'
entanglement_gates = 'cz'
layers = 1
entanglement = 'linear'

molecule = MoleculeInfo(
        #Coordinates in Angstrom
        symbols=["Li", "H"],
        coords=([0.0, 0.0, 0.0], [1.57, 0.0, 0.0]),
        multiplicity=1,  # = 2*spin + 1
        charge=0
)
taper='JordanWigner'
freezecore=2

#graph = nx.random_regular_graph(3, number_of_qubits, seed=seed)
#w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))

w = np.array([[0, 1 ,1], [1, 0 , 1], [1, 1, 0]])

problem = {'type':'MaxCut', 'properties': w}

#my_molecule=Moleculeclass(molecule,'Parity',2)
#print(my_molecule.get_qubit_operator())

# aavqechem=AAVQE_on_Chemistry(molecule, taper,freezecore,steps, layers, single_qubit_gates, entanglement_gates,entanglement)
# aavqechem.run()
# print(aavqechem.minimum_eigenvalue())

Solvebynumpy(molecule).run()
#np.random.seed(2)
#Brute_Force(problem)

#aavqe = AAVQE(number_of_qubits, problem, steps, layers, single_qubit_gates, entanglement_gates, entanglement)
#aavqe.run()

#aqc_pqc = AQC_PQC_QAOA(number_of_qubits, problem, steps, layers, use_null_space=True) #Uncomment if you want to use QAOA ansatz.
#aqc_pqc.run()

#aqc_pqc = AQC_PQC(number_of_qubits, problem, steps, layers, single_qubit_gates,
#                  entanglement_gates, entanglement, use_null_space=True, use_third_derivatives=False)
#aqc_pqc.run()
