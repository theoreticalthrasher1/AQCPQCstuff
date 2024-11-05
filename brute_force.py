import numpy as np
import collections


class Brute_Force():
    def __init__(self, problem):

        self.problem_type = problem['type']
        self.problem_properties = problem['properties']

        optimal_cost, optimal_strings = self.get_optimal_cost_and_strings()
        print(f'For the given {self.problem_type} problem the optimal cost function value is {optimal_cost} and the optimal bitstrings are {optimal_strings}')


    def get_optimal_cost_and_strings(self):

        #Note that in MaxCut we are searching for the maximum cost function value while in Number Paritioning for the minimum.


        if self.problem_type == 'MaxCut': #The following code was taken from qiskit textbook!

            adjacency_matrix = self.problem_properties
            optimal_cost = 0
            number_of_qubits = len(adjacency_matrix)
            cost_function_values = collections.defaultdict(list)
            for b in range(2**number_of_qubits):
                x = [int(t) for t in reversed(list(bin(b)[2:].zfill(number_of_qubits)))]
                cost = 0
                for i in range(number_of_qubits):
                    for j in range(number_of_qubits):
                        cost += adjacency_matrix[i,j] * x[i] * (1-x[j])

                cost = np.round(cost, 7)
                x.reverse()
                cost_function_values[cost].append(x)

                if optimal_cost < cost:
                    optimal_cost = cost

            cost_function_values = sorted(cost_function_values.items())
            optimal_strings = cost_function_values[-1][1]


        elif self.problem_type == 'NumberPartitioning':

            numbers_list = self.problem_properties
            best_cost = np.inf
            number_of_qubits = len(numbers_list)
            cost_function_values = collections.defaultdict(list)

            for b in range(2**number_of_qubits):
                x = [int(t) for t in reversed(list(bin(b)[2:].zfill(number_of_qubits)))]
                cost = 0
                for i in range(number_of_qubits):
                    cost += (2*x[i]-1)*numbers_list[i]


                cost = cost**2
                cost = np.round(cost, 7)

                x.reverse()
                cost_function_values[cost].append(x)

                if best_cost > cost:
                    best_cost = cost

            cost_function_values = sorted(cost_function_values.items())
            optimal_strings = cost_function_values[0][1]

        return optimal_cost, optimal_strings
            