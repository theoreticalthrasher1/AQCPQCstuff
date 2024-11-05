from qiskit import QuantumCircuit
from qiskit.visualization import *
from qiskit.circuit.library import TwoLocal, QAOAAnsatz
import numpy as np
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

class QAOA_Circuit():
    def __init__(self, number_of_qubits, layers, problem, angles):

        self.number_of_qubits = number_of_qubits
        self.qcir = QuantumCircuit(self.number_of_qubits)
        self.layers = layers
        self.problem = problem
        self.angles = angles #In QAOA we apply the problem and mixer unitaries alternatively. We feed the angles in the form [gamma_0, beta_0, ..., gamma_p, beta_p]
        self.gammas = [self.angles[2*p] for p in range(self.layers)]
        self.betas = [self.angles[2*p + 1] for p in range(self.layers)]

        self.qcir.h(range(self.number_of_qubits)) #Initialize quantum circuit in the |+> state

        self.qcir &= self.get_qaoa_ansatz(self.problem, self.layers, self.betas, self.gammas)
        #self.qcir.draw(output='mpl')
        #plt.show()
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
