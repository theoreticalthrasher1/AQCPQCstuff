from qiskit import QuantumCircuit
from qiskit.visualization import *
from qiskit.circuit.library import TwoLocal, QAOAAnsatz
import numpy as np
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt


class QCir():
    def __init__(self, number_of_qubits, thetas, layers, single_qubit_gates, entanglement_gates, entanglement, use_QAOA = False, problem = None, betas=None, gammas=None):

        self.number_of_qubits = number_of_qubits
        self.qcir = QuantumCircuit(self.number_of_qubits)
        self.single_qubit_gates = single_qubit_gates
        self.entanglement_gates = entanglement_gates
        self.layers = layers
        self.use_QAOA = use_QAOA
        self.problem = problem
        self.betas = betas
        self.gammas = gammas

        if not self.use_QAOA:
            self.qcir &= TwoLocal(num_qubits = number_of_qubits, rotation_blocks = single_qubit_gates, entanglement_blocks = entanglement_gates, reps = layers, entanglement=entanglement)
            self.number_of_parameters = self.qcir.num_parameters 

            if thetas == 'initial':
                thetas = self.get_initial_parameters()

            self.qcir = self.qcir.assign_parameters(thetas)

        else:
            self.qcir += self.get_qaoa_ansatz(self.problem, self.layers, self.betas, self.gammas)
            self.qcir.draw(output='mpl')
            plt.show()
            self.number_of_parameters = 2*self.layers
            


    def get_qaoa_ansatz(self, problem, layers, betas, gammas):
        
        quantum_circuit = QuantumCircuit(self.number_of_qubits)

        if problem['type'] == 'MaxCut':
            w = problem['properties']

            for layer in range(layers):
                for qubit1 in range(self.number_of_qubits):
                    for qubit2 in range(self.number_of_qubits):
                        if qubit1 < qubit2 and w[qubit1, qubit2] != 0:
                            quantum_circuit.rzz(gammas[layer]*w[qubit1, qubit2], qubit1, qubit2)

                quantum_circuit.barrier()

                for qubit in range(self.number_of_qubits):
                    quantum_circuit.rx(betas[layer], qubit)
                    
        
        elif problem['type'] == 'NumberPartitioning':
            numbers_list = problem['properties']


            for layer in range(layers):
                for qubit1 in range(self.number_of_qubits):
                    for qubit2 in range(self.number_of_qubits):
                        if qubit1 < qubit2:
                            quantum_circuit.rzz(gammas[layer]*numbers_list[qubit1]*numbers_list[qubit2], qubit1, qubit2)

                quantum_circuit.barrier()

                for qubit in range(self.number_of_qubits):
                    quantum_circuit.rx(betas[layer], qubit)

        return quantum_circuit

    def get_initial_parameters(self):

        initial_parameters = []

        if self.single_qubit_gates == 'ry' and self.entanglement_gates == 'cz':
            for qubit in range(self.number_of_qubits*(self.layers)):
                initial_parameters.append(0)

            for qubit in range(self.number_of_qubits):
                initial_parameters.append(np.pi/2)

        elif self.single_qubit_gates == ['rx', 'ry'] and self.entanglement_gates == 'cz':
            for qubit in range((2*self.layers+1)*self.number_of_qubits):
                initial_parameters.append(0)

            for qubit in range(self.number_of_qubits):
                initial_parameters.append(np.pi/2)

        return initial_parameters 
    
    def calculate_expectation_value(self, matrix): #This function calculates the expectation value of a given observable (given as a matrix)
        statevector = Statevector.from_label('0'*self.number_of_qubits)
        statevector = statevector.evolve(self.qcir)
        expectation_value = statevector.expectation_value(matrix)
        return expectation_value
