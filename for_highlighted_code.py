def alternative_run(self):
       lambdas = [i for i in np.linspace(0, 1, self.steps+1)][1:]

        optimal_thetas = self.initial_parameters.copy()
        instantaneous_expectation_value=self.get_expectation_value(optimal_thetas,self.initial_hamiltonian)
        initial_ground_state=self.minimum_eigenvalue(self.initial_hamiltonian)
        energies_aavqe = [instantaneous_expectation_value]
        energies_exact = [initial_ground_state]
        #Do a pre-run of the initial angles. Fix the initial Hamiltonian and have it run VQE to get the correct angles to start with. 
        # minimization_object = optimize.minimize(self.get_expectation_value, x0=optimal_thetas, args=(self.initial_hamiltonian), method='SLSQP')
        # optimal_thetas = minimization_object.x
        #that didn't work. Input it manually. It outputs IIIZIIIZ, so it's |00010001>. 

        
        print(f'We start with the optimal angles of the initial hamiltonian: {optimal_thetas}')


        for lamda in lambdas:

            print('\n')
            hamiltonian = self.get_instantaneous_hamiltonian(lamda)

            minimization_object = optimize.minimize(self.get_expectation_value, x0=optimal_thetas, args=(hamiltonian), method='SLSQP')
            optimal_thetas = minimization_object.x
            print(f'We are working on {lamda} where the current optimal point is {optimal_thetas}')

            self.offset=0

            inst_exp_value = self.get_expectation_value(optimal_thetas, hamiltonian) - lamda*self.offset
            energies_aavqe.append(inst_exp_value)
            energies_exact.append(self.minimum_eigenvalue(hamiltonian) - lamda*self.offset)
            #print(f'and the hamiltonian right now is {hamiltonian} ')
            
            print(f'and the instantaneous expectation values is {inst_exp_value}') 
            print(f'and the true expectation value is {self.minimum_eigenvalue(hamiltonian) - lamda*self.offset}')
#Question now is how will we compute the true expectation value? Will we do it from the Hamiltonian that was created? 

        plt.plot(energies_aavqe,label='aavqe energy')
        plt.plot(energies_exact,label='true energy')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('energy (Ha)')
        plt.title(f'{self.string_initial_hamiltonian} and {self.string_final_hamiltonian}')
        plt.show()
        return energies_aavqe
