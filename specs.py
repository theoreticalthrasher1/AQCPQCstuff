from ManualOperator import IBM_LiH, IBM_LiH_initial
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from Quantum_Chemistry import Moleculeclass

#Equilibrium bond length and the bond length explored in IBM paper both given:
# distance=1.57
distance=2.5
hamiltonian_methods = {
    'initial': {
        'qiskit_hf': {
            'description': 'Use Qiskit Hartree-Fock method to generate Initial Hamiltonian',
            'generate': lambda molecule, taper, freezecore: Moleculeclass(molecule, taper, freezecore).get_hartreefock_energy()* Moleculeclass(molecule, taper, freezecore).get_hartreefock_in_pauli()
        },
        'paper': {
            'description': 'Use Hartree-Fock from a specific paper (custom implementation)',
            'generate': lambda molecule, taper, freezecore:IBM_LiH_initial
        }
    },
    'final': {
        'qiskit_method ': {
            'description': 'Use final Hamiltonian from a paper-specific method',
            'generate': lambda molecule, taper, freezecore: Moleculeclass(molecule, taper,freezecore).get_qubit_operator()
        },
        'paper': {
            'description': 'Use Qiskit for final Hamiltonian (if different from initial)',
            'generate': lambda molecule, taper, freezecore: IBM_LiH
        }
    }
}
molecule = MoleculeInfo(
        #Coordinates in Angstrom
        symbols=["Li", "H"],
        coords=([0.0, 0.0, 0.0], [distance, 0.0, 0.0]),
        multiplicity=1,  # = 2*spin + 1
        charge=0
)
taper='JordanWigner'
freezecore=2