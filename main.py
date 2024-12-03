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
#from qiskit_aer import Aer
#chemical accuracy
#backend = Aer.get_backend('statevector_simulator')

#We've put a minus sign in front of the initial hf hamiltonian, 
#final hamiltonian might be wrong? Also, we don't need an offset. 
#We need to change the initial angles each time we choose a different initial hamiltonian. 
#We also need to figure out angles for the LiH examples.
#The energies were increasing linearly
#Are we actually getting the correct qubit
#remove unnecessary stuff, and make sure everything works on basic examples, then we can scale it up. 


# #seeds=[20, 21, 30, 33, 36, 42, 43, 55, 67,170 ]
# seeds=67
# algorithm_globals.random_seed= seeds
# seed_transpiler = seeds
# iterations = 125
# shot = 6000

seed = 3
number_of_qubits = 4
steps = 10#Choose number of steps to interpolate from initial to final Hamiltonian
connectivity = 'nearest-neighbors' #This is the connectivity of the non-parameterized gates in the Hardware-Efficient ansatz
single_qubit_gates = 'ry'
entanglement_gates = 'cz'
layers = 1
entanglement = 'linear'




#aavqechem=AAVQE_on_Chemistry(molecule, taper,freezecore,steps, layers, single_qubit_gates, entanglement_gates,entanglement)
#hf=Moleculeclass(molecule,taper,freezecore).get_hartreefock_in_pauli()
#print(hf)
hfstate=Moleculeclass(molecule,taper,freezecore).get_hartreefock_in_pauli()
#print(hfstate)
#print(hfstate)
#myaavqe=My_AAVQE(number_of_qubits,steps,layers,single_qubit_gates,entanglement_gates,entanglement,hf,qubitop)
myaavqe=My_AAVQE(number_of_qubits,steps,layers,single_qubit_gates,entanglement_gates,entanglement,'paper','paper',hfstate)
#print(myaavqe.initial_hamiltonian)
#H=myaavqe.initial_hamiltonian
# Assuming `H` is your operator
#matrix = H.to_matrix()

# Compute eigenvalues and eigenvectors using numpy.linalg.eig

# Print the minimum eigenvalue and corresponding eigenvector

#myaavqe.initial_hamiltonian()
#print(myaavqe.draw_latex())
# Print the non-zero elements and their positions
myaavqe.alternative_run()
#print(myaavqe.draw_latex())



#print(myaavqe.get_instantaneous_hamiltonian(1))
#print(aavqechem.minimum_eigenvalue())

#Solvebynumpy(molecule).run()
#np.random.seed(2)
#Brute_Force(problem)
#aavqe = AAVQE(number_of_qubits, problem, steps, layers, single_qubit_gates, entanglement_gates, entanglement)
#aavqe.run()

#aqc_pqc = AQC_PQC_QAOA(number_of_qubits, problem, steps, layers, use_null_space=True) #Uncomment if you want to use QAOA ansatz.
#aqc_pqc.run()

#aqc_pqc = AQC_PQC(number_of_qubits, problem, steps, layers, single_qubit_gates,
#                  entanglement_gates, entanglement, use_null_space=True, use_third_derivatives=False)
#aqc_pqc.run()
#print(hf)
#myaavqe.run()

#For calculating the ground state energy manually:
# mat=IBM_LiH.to_matrix()
# print(np.min(np.linalg.eig(mat)[0]))



# Graph specifications:
# #graph = nx.random_regular_graph(3, number_of_qubits, seed=seed)
# #w = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes()))

# w = np.array([[0, 1 ,1], [1, 0 , 1], [1, 1, 0]])

# problem = {'type':'MaxCut', 'properties': w}