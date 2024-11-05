from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver


class Moleculeclass():
    def __init__(self, molecule, taper,freezecore):
        self.molecule = molecule
   
        self.taper = taper
        self.freezecore=freezecore
        self.electronic_structure_problem = PySCFDriver.from_molecule(self.molecule).run()
        n=0
        while n<freezecore:
            self.electronic_structure_problem=FreezeCoreTransformer().transform(self. electronic_structure_problem) 
            n+=1
        second_quantized_operators =  self.electronic_structure_problem.second_q_ops()
        second_quantized_operators=second_quantized_operators[0]
   

        if self.taper == 'JordanWigner':
            self.qubit_operator = JordanWignerMapper().map(second_quantized_operators)

        elif self.taper == 'Parity':
            self.qubit_operator = ParityMapper().map(second_quantized_operators)
        else:
            raise ValueError("Unsupported tapering method. Choose 'JordanWigner' or 'Parity'.")

    # Method to get the qubit operator if needed outside
   
    def get_qubit_operator(self):
        return self.qubit_operator
    
molecule = MoleculeInfo(
        #Coordinates in Angstrom
        symbols=["Li", "H"],
        coords=([0.0, 0.0, 0.0], [1.57, 0.0, 0.0]),
        multiplicity=1,  # = 2*spin + 1
        charge=0
)


class Solvebynumpy():
    def __init__(self,molecule):
        self.molecule=molecule
        self.electronic_structure_problem = PySCFDriver.from_molecule(self.molecule).run()
    def run(self):
        numpy_solver = NumPyMinimumEigensolver()
        mapper = JordanWignerMapper()
        calc = GroundStateEigensolver(mapper, numpy_solver)
        res = calc.solve(self.electronic_structure_problem)
        print(res)


   
# class Moleculeclass():
#     def __init__(self,molecule,taper):
#         self.molecule=molecule
#         self.taper=taper
#         electronic_structure_problem=PySCFDriver.from_molecule(self.molecule).run()
#         second_quantized_operators=electronic_structure_problem.second_q_ops()
#         if self.taper=='JordanWigner':
#             return JordanWignerMapper().map(second_quantized_operators)
#         else self.taper=='Parity'
#             return ParityMapper().map(second_quantized_operators)
#     # def Hamiltonian(self,freezecore=False):
#     #     driver=PySCFDriver.from_molecule(self.molecule)
#     #     qmolecule=driver.run()
#     #     self.freeze=freezecore
#     #     if not self.freeze: 
#     #         hamiltonian=qmolecule.hamiltonian
#     #         return hamiltonian
#     #     else:
#     #         transformer = FreezeCoreTransformer()
#     #         qmolecule_frozen=transformer.transform(qmolecule)
#     #         hamiltonian=qmolecule_frozen.hamiltonian
#     #         return hamiltonian
#     def Qubit_Hamiltonian(self, Pauli_Map_Type,freezecore=False):
#         driver=PySCFDriver.from_molecule(self.molecule)
#         qmolecule=driver.run()
#         self.Pauli=Pauli_Map_Type
#         self.freeze=freezecore
#         if not self.freeze:
#             hamiltonian=qmolecule.hamiltonian
#             second_q_op=hamiltonian.second_q_op()
#             return second_q_op
#         else:
#             transformer =   ()
#             qmolecule_frozen=transformer.transform(qmolecule)
#             qmolecule_frozen=transformer.transform(qmolecule)
#             hamiltonian=qmolecule_frozen.hamiltonian
#             second_q_op=hamiltonian.second_q_op()
#             if self.Pauli==None:
#                     return second_q_op
#             elif self.Pauli=='JordanWigner':
#                 return JordanWignerMapper().map(second_q_op)
#             elif self.Pauli=='Parity':
#                 return ParityMapper(num_particles=qmolecule.num_particles).map(second_q_op)
#         def Taper_me(self):
            




# #How can I nicely define the tapering and stuff? 
#     def Solve(self):
#         driver=PySCFDriver.from_molecule(self.molecule)
#         qmolecule=driver.run()
#         solver = GroundStateEigensolver(
#         JordanWignerMapper(),
#         NumPyMinimumEigensolver(),
#         )
#         result = solver.solve(qmolecule)
#         return result
#     def Ref_Value(self):
#         driver=PySCFDriver.from_molecule(self.molecule)
#         qmolecule=driver.run()
#         solver = GroundStateEigensolver(
#         JordanWignerMapper(),
#         NumPyMinimumEigensolver(),
#         )
#         result = solver.solve(qmolecule)
#         ref_value = result.computed_energies + result.nuclear_repulsion_energy
#         return ref_value


# #No need for the Z2 symmetries since the tepered map already exploits these            
        

            


# qmolecule = driver.run()
# hamiltonian = qmolecule.hamiltonian
# coefficients = hamiltonian.electronic_integrals
# #print(coefficients.alpha)



#second_q_op = hamiltonian.second_q_op()
# print('qmolecule.num_spatial_orbitals:',qmolecule.num_spatial_orbitals)
# print('qmolecule.num_particles:',qmolecule.num_particles)
# #print(second_q_op)

# properties = driver.run()
# # Now you can get the reduced electronic structure problem
# qmolecule = FreezeCoreTransformer(
#     freeze_core=True, remove_orbitals=[-3, -2]
# ).transform(properties)
