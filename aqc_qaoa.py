from qiskit.visualization import *
import numpy as np
import scipy.optimize as optimize
import networkx as nx
import collections
from hamiltonian import Hamiltonian
from qaoa_circuit import QAOA_Circuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from sympy import Matrix


class AQC_PQC_QAOA():
    def __init__(self, number_of_qubits, problem, steps, layers, use_null_space = False):

        self.number_of_qubits = number_of_qubits
        self.problem = problem
        self.steps = steps
        self.layers = layers
        self.initial_parameters = [0 for _ in range(2*self.layers)]
        self.number_of_parameters = 2*self.layers
        self.use_null_space = use_null_space

        hamiltonians = Hamiltonian(self.problem)

        self.target_hamiltonian, self.offset = hamiltonians.get_pauli_operator_and_offset()
        self.initial_hamiltonian = hamiltonians.get_transverse_hamiltonian(self.number_of_qubits)


    def get_expectation_value(self, angles, observable):
        circuit = QAOA_Circuit(self.number_of_qubits, self.layers, self.problem, angles)
        sv1 = Statevector.from_label('0'*self.number_of_qubits)
        sv1 = sv1.evolve(circuit.qcir)
        expectation_value = sv1.expectation_value(observable)
        return np.real(expectation_value)
    

    def get_derivative(self, observable, which_parameter, parameters, epsilon=0.000001):

        derivative = 0
        parameters_plus, parameters_minus = parameters.copy(), parameters.copy()
        parameters_plus[which_parameter] += epsilon
        parameters_minus[which_parameter] -= epsilon

        derivative += 1/2*self.get_expectation_value(parameters_plus, observable)
        derivative -= 1/2*self.get_expectation_value(parameters_minus, observable)
        derivative /= epsilon

        return derivative


    def get_hessian_matrix(self, observable, angles, epsilon=0.000001):

        hessian = np.zeros((self.number_of_parameters, self.number_of_parameters))
    
        for parameter1 in range(self.number_of_parameters):
            for parameter2 in range(self.number_of_parameters):
                if parameter1 <= parameter2:    
                    
                    hessian_thetas_1, hessian_thetas_2, hessian_thetas_3, hessian_thetas_4 = angles.copy(), angles.copy(), angles.copy(), angles.copy()

                    hessian_thetas_1[parameter1] += epsilon
                    hessian_thetas_1[parameter2] += epsilon


                    hessian_thetas_2[parameter1] -= epsilon
                    hessian_thetas_2[parameter2] += epsilon

                    hessian_thetas_3[parameter1] += epsilon
                    hessian_thetas_3[parameter2] -= epsilon

                    hessian_thetas_4[parameter1] -= epsilon
                    hessian_thetas_4[parameter2] -= epsilon

                    hessian[parameter1, parameter2] += self.get_expectation_value(hessian_thetas_1, observable)/4
                    hessian[parameter1, parameter2] -= self.get_expectation_value(hessian_thetas_2, observable)/4
                    hessian[parameter1, parameter2] -= self.get_expectation_value(hessian_thetas_3, observable)/4
                    hessian[parameter1, parameter2] += self.get_expectation_value(hessian_thetas_4, observable)/4
                    hessian[parameter1, parameter2] /= (epsilon**2)

                    hessian[parameter2, parameter1] = hessian[parameter1, parameter2]
                    

        return hessian

    

    def get_instantaneous_hamiltonian(self, time):
        return (1-time)*self.initial_hamiltonian + time*self.target_hamiltonian
    
    def get_linear_system(self, hamiltonian, angles): 

        zero_order_terms = np.zeros((self.number_of_parameters,))
        first_order_terms = np.zeros((self.number_of_parameters, self.number_of_parameters))

        for parameter in range(self.number_of_parameters):
            zero_order_terms[parameter] += self.get_derivative(hamiltonian, parameter, angles)

        first_order_terms = self.get_hessian_matrix(hamiltonian, angles)

        return np.array(zero_order_terms), np.array(first_order_terms)
    
    def find_indices(self, s, threshold=0.1):
        indices = []
        for k in range(self.number_of_parameters):
            if s[k] <= threshold:
                indices.append(k)

        return indices

    def minimum_eigenvalue(self, matrix):

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        min_eigen = np.min(eigenvalues)
        print(min_eigen)

        return min_eigen
        #return np.min(np.linalg.eig(matrix)[0])

    def run(self):
        
        energies_aqcpqc = []

        lambdas = [i for i in np.linspace(0, 1, self.steps+1)][1:]
        optimal_thetas = self.initial_parameters.copy()
        print(f'We start with the optimal angles of the initial hamiltonian: {optimal_thetas}')

        initial_hessian = self.get_hessian_matrix(self.initial_hamiltonian, optimal_thetas) 
        w, v = np.linalg.eig(initial_hessian)
        print(f'The eigenvalues of the initial Hessian are {np.round(w, 7)}')

        for lamda in lambdas:
            print('\n')
            print(f'We are working on {lamda} where the current optimal point is {optimal_thetas}')
            hamiltonian = self.get_instantaneous_hamiltonian(lamda)
            zero, first = self.get_linear_system(hamiltonian, optimal_thetas)
            print(zero)

            hessian = self.get_hessian_matrix(hamiltonian, optimal_thetas)
            print(f'The eigs of Hessian are {np.linalg.eig(first)[0]}')


            def equations(x):
                array = np.array([x[_] for _ in range(self.number_of_parameters)])
                equations = zero + first@array

                y = np.array([equations[_] for _ in range(self.number_of_parameters)])
                return y@y
            
            def norm(x):
                return np.linalg.norm(x)
            

            if not self.use_null_space:


                def minim_eig_constraint(x):
                    new_thetas = [optimal_thetas[i] + x[i] for i in range(self.number_of_parameters)]
                    return self.minimum_eigenvalue(self.get_hessian_matrix(hamiltonian, new_thetas))

                cons = [{'type': 'ineq', 'fun':minim_eig_constraint}]

                res = optimize.minimize(equations, x0 = [0 for _ in range(self.number_of_parameters)], constraints=cons,  method='SLSQP',  options={'disp': False, 'maxiter':700}) 
                epsilons = [res.x[_] for _ in range(self.number_of_parameters)]
            
            
                print(f'The solutions of equations are {epsilons}')
                optimal_thetas = [optimal_thetas[_] + epsilons[_] for _ in range(self.number_of_parameters)]

            else:
                                
                                
                u, s, v = np.linalg.svd(first)
                indices = self.find_indices(s)
                print(f'The singular values of matrix A are {s}')

                null_space_approx = [v[index] for index in indices]

                cons = [{'type': 'eq', 'fun':equations}]
                unconstrained_optimization = optimize.minimize(norm, x0 = [0 for _ in range(self.number_of_parameters)], constraints=cons, method='SLSQP',  options={'disp': False})
                epsilons_0 = unconstrained_optimization.x


                print(f'A solution to the linear system of equations is {epsilons_0}')
                optimal_thetas = [optimal_thetas[i] + epsilons_0[i] for i in range(self.number_of_parameters)]

                
                def norm(x):
                    vector = epsilons_0.copy()
                    for _ in range(len(null_space_approx)):
                        vector += x[_]*null_space_approx[_]

                    norm = np.linalg.norm(vector)
                    #print(f'Norm: {norm}')
                    return norm
                

                def minim_eig_constraint(x):
                    new_thetas = optimal_thetas.copy()
                    for _ in range(len(null_space_approx)):
                        new_thetas += x[_]*null_space_approx[_]
                    return self.minimum_eigenvalue(self.get_hessian_matrix(hamiltonian, new_thetas))
                    

                cons = [{'type': 'ineq', 'fun':minim_eig_constraint}]
                constrained_optimization = optimize.minimize(norm, x0=[0 for _ in range(len(null_space_approx))], constraints=cons, method='COBYLA', options={'disp':True, 'maxiter':400}) 
                print(f'The solutions of the second optimization are {constrained_optimization.x}')

                for _ in range(len(null_space_approx)):
                    optimal_thetas += constrained_optimization.x[_]*null_space_approx[_]



            hessian = self.get_hessian_matrix(hamiltonian, optimal_thetas)
            min_eigen = self.minimum_eigenvalue(hessian)


            inst_exp_value = self.get_expectation_value(optimal_thetas, hamiltonian) - lamda*self.offset
            energies_aqcpqc.append(inst_exp_value)

            print(f'and the minimum eigenvalue of the Hessian at the solution is {min_eigen}')
            print(f'and the instantaneous expectation values is {inst_exp_value}') 
            print(f'and the exact minimum energy is {self.minimum_eigenvalue(hamiltonian) - lamda*self.offset}')



        return energies_aqcpqc

