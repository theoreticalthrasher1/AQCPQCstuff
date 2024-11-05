import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.applications import Maxcut, NumberPartition

class Hamiltonian():


    def __init__(self, problem):
        self.problem_type = problem['type']
        self.problem_properties = problem['properties']


    def get_pauli_operator_and_offset(self):
        
        if self.problem_type == 'MaxCut':
            
            w = self.problem_properties
            maxcut = Maxcut(w)
            qp = maxcut.to_quadratic_program()
            qubitOp, offset = qp.to_ising()


        elif self.problem_type == 'Number_Partition':
            
            number_list = self.problem_properties
            num_par = NumberPartition(number_list)
            qp = num_par.to_quadratic_program()
            qubitOp, offset = qp.to_ising()

        return qubitOp, offset
    
    def get_transverse_hamiltonian(self, number_of_qubits, alternative=False):
        X_tuples = []

        if not alternative:
            
            for i in range(number_of_qubits):
                X_tuples.append(('X', [i], -1))

            hamiltonian = SparsePauliOp.from_sparse_list([*X_tuples], num_qubits = number_of_qubits)

        else:
            
            ids_tuples = []
            coefs = [np.random.choice([-1/4, -3/4]) for _ in range(number_of_qubits)]

            for i in range(number_of_qubits):
                X_tuples.append(('X', [i], coefs[i]))
                ids_tuples.append(('I', [i], -coefs[i]))


            hamiltonian = SparsePauliOp.from_sparse_list([*X_tuples, *ids_tuples], num_qubits = number_of_qubits)
        
        return hamiltonian


    def seans_initial_hamiltonian(self):
        pass

    def seans_target_hamiltonian(self):
        pass

    def seans_hamiltonian(self, lamda):



        pauli_string = ('Z', [0], -1)
        pauli_string2 = ('X', [0], lamda)

        Hamiltonian = SparsePauliOp.from_sparse_list([pauli_string, pauli_string2], num_qubits = 1)

        return Hamiltonian