from qiskit.visualization import *
import numpy as np
import scipy.optimize as optimize
from hamiltonian import Hamiltonian
from quantum_circuit import QCir
from qiskit.primitives import Estimator, StatevectorEstimator
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
import matplotlib.pyplot as plt
from Quantum_Chemistry import *
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

class AAVQE_on_Chemistry():
    def __init__(self, molecule, taper,freezecore,steps, layers, single_qubit_gates, entanglement_gates,entanglement):
        self.molecule= molecule
        self.taper= taper
        self.freezecore= freezecore
        self.steps= steps
        self.layers= layers
        self.single_qubit_gates= single_qubit_gates
        self.entanglement_gates= entanglement_gates
        self.entanglement=entanglement
        self.qubitop=Moleculeclass(molecule, taper,freezecore).get_qubit_operator()
        self.number_of_qubits=self.qubitop.num_qubits
    def run(self):
        aavqe_instance=My_AAVQE(self.number_of_qubits,self.steps,self.layers,self.single_qubit_gates,self.entanglement_gates,self.entanglement,'transverse',self.qubitop)
        return aavqe_instance.run()
    def groundstateeigsolver(self):
        solver= GroundStateEigensolver(JordanWignerMapper(),NumPyMinimumEigensolver())
        
        return solver.solve(self.elecprob)
    def minimum_eigenvalue(self):
        min_eigen = np.min(np.linalg.eig(self.qubitop)[0])
        return min_eigen
'''

solver = GroundStateEigensolver(
    JordanWignerMapper(),
    NumPyMinimumEigensolver(),
)

result = solver.solve(qmolecule)
print(result.computed_energies)
print(result.nuclear_repulsion_energy)
ref_value = result.computed_energies + result.nuclear_repulsion_energy
print(ref_value)
'''



class My_AAVQE():
    def __init__(self, number_of_qubits, steps, layers, single_qubit_gates, entanglement_gates, entanglement,initial_hamiltonian,target_hamiltonian):
        self.number_of_qubits = number_of_qubits    
        self.steps = steps
        self.initial_hamiltonian = initial_hamiltonian
        self.target_hamiltonian=target_hamiltonian
        self.offset=0
        self.layers = layers
        self.single_qubit_gates = single_qubit_gates
        self.entanglement_gates = entanglement_gates
        self.entanglement = entanglement
        # Dealing with the initial hamiltonian
        if initial_hamiltonian == 'transverse':
            X_tuples = []

            for i in range(number_of_qubits):
                X_tuples.append(('X', [i], -1))

            self.initial_hamiltonian = SparsePauliOp.from_sparse_list([*X_tuples], num_qubits = number_of_qubits)
        else:

            self.initial_hamiltonian  = initial_hamiltonian
        # Now setting up the quantum circuit. 


        self.initial_parameters = QCir(self.number_of_qubits,'initial' ,self.layers, self.single_qubit_gates, self.entanglement_gates, self.entanglement).get_initial_parameters()
        self.qcir = TwoLocal(self.number_of_qubits, self.single_qubit_gates, self.entanglement_gates, self.entanglement, self.layers)
        #this is already in the general parameter form. 
        self.number_of_parameters = len(self.initial_parameters)


    def draw_circuit(self):
        self.qcir.decompose().draw(output='mpl')
        plt.show()

    def get_expectation_value(self, angles, observable):
        estimator = StatevectorEstimator()
        pub = (self.qcir, observable, angles)
        job = estimator.run([pub])
        
        result = job.result()[0]
        expectation_value = result.data.evs

        return np.real(expectation_value)
    
    
    def get_derivatives(self, angles, observable, shots=None): 
        estimator = Estimator() if not shots else Estimator(options={'shots':shots})
   
        gradient = ParamShiftEstimatorGradient(estimator)
        derivatives = gradient.run(self.qcir, observable, [angles]).result().gradients[0]
        return derivatives
    def get_instantaneous_hamiltonian(self, time):
        return (1-time)*self.initial_hamiltonian + time*self.target_hamiltonian




    def minimum_eigenvalue(self, matrix):

        min_eigen = np.min(np.linalg.eig(matrix)[0])
       
        #print(min_eigen)
        return min_eigen

    def run(self):
        
        energies_aavqe = []

        lambdas = [i for i in np.linspace(0, 1, self.steps+1)][1:]
        
    
        optimal_thetas = self.initial_parameters.copy()
        print(f'We start with the optimal angles of the initial hamiltonian: {optimal_thetas}')


        for lamda in lambdas:

            print('\n')
            print(f'We are working on {lamda} where the current optimal point is {optimal_thetas}')
            hamiltonian = self.get_instantaneous_hamiltonian(lamda)

            minimization_object = optimize.minimize(self.get_expectation_value, x0=optimal_thetas, args=(hamiltonian), method='SLSQP')
            optimal_thetas = minimization_object.x


            inst_exp_value = self.get_expectation_value(optimal_thetas, hamiltonian) - lamda*self.offset
            energies_aavqe.append(inst_exp_value)
            #print(f'and the hamiltonian right now is {hamiltonian} ')
            print(f'and the instantaneous expectation values is {inst_exp_value}') 
            print(f'and the true expectation value is {self.minimum_eigenvalue(hamiltonian) - lamda*self.offset}')
#Question now is how will we compute the true expectation value? Will we do it from the Hamiltonian that was created? 

        return energies_aavqe




class AAVQE():
    def __init__(self, number_of_qubits, problem, steps, layers, single_qubit_gates, entanglement_gates, entanglement):

        self.number_of_qubits = number_of_qubits
        self.problem = problem
        self.steps = steps
        self.layers = layers
        self.single_qubit_gates = single_qubit_gates
     
        self.entanglement_gates = entanglement_gates
        self.entanglement = entanglement

        qcir = QCir(self.number_of_qubits, 'initial', self.layers, self.single_qubit_gates, self.entanglement_gates, self.entanglement)

        self.initial_parameters = qcir.get_initial_parameters()
        self.number_of_parameters = len(self.initial_parameters)

        hamiltonians = Hamiltonian(self.problem)

     
        self.target_hamiltonian, self.offset = hamiltonians.get_pauli_operator_and_offset()
        self.initial_hamiltonian = hamiltonians.get_transverse_hamiltonian(self.number_of_qubits)   
        self.qcir = TwoLocal(self.number_of_qubits, self.single_qubit_gates, self.entanglement_gates, self.entanglement, self.layers)
        self.qcir.decompose().draw(output='mpl')
        plt.show()
    def get_expectation_value(self, angles, observable):
        estimator = StatevectorEstimator()
       
       
        pub = (self.qcir, observable, angles)
        job = estimator.run([pub])
        result = job.result()[0]
        expectation_value = result.data.evs

        return np.real(expectation_value)
    
    
    
    def get_derivatives(self, angles, observable, shots=None):

        estimator = Estimator() if not shots else Estimator(options={'shots':shots})
        gradient = ParamShiftEstimatorGradient(estimator)

        derivatives = gradient.run(self.qcir, observable, [angles]).result().gradients[0]

        return derivatives


    def get_instantaneous_hamiltonian(self, time):
        return (1-time)*self.initial_hamiltonian + time*self.target_hamiltonian



    def minimum_eigenvalue(self, matrix):

        min_eigen = np.min(np.linalg.eig(matrix)[0])
        print(min_eigen)
        return min_eigen

    def run(self):
        
        energies_aavqe = []

        lambdas = [i for i in np.linspace(0, 1, self.steps+1)][1:]
        optimal_thetas = self.initial_parameters.copy()
        print(f'We start with the optimal angles of the initial hamiltonian: {optimal_thetas}')


        for lamda in lambdas:

            print('\n')
            print(f'We are working at t={lamda} where the current optimal point is {optimal_thetas}')
            hamiltonian = self.get_instantaneous_hamiltonian(lamda)

            minimization_object = optimize.minimize(self.get_expectation_value, x0=optimal_thetas, args=(hamiltonian), method='SLSQP')
            optimal_thetas = minimization_object.x


            inst_exp_value = self.get_expectation_value(optimal_thetas, hamiltonian) - lamda*self.offset
            energies_aavqe.append(inst_exp_value)

            print(f'and the instantaneous expectation values is {inst_exp_value}') 
            print(f'and the true expectation value is {self.minimum_eigenvalue(hamiltonian) - lamda*self.offset}')

        return energies_aavqe