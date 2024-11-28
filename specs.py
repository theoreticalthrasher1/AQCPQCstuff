from ManualOperator import IBM_LiH, IBM_LiH_initial
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from Quantum_Chemistry import Moleculeclass

#Equilibrium bond length and the bond length explored in IBM paper both given:
distance=1.57
#distance=2.5
hamiltonian_methods = {
    'initial': {
        'qiskit_hf': {
            'description': 'This is a simple Hartree-Fock Hamiltonian.',
            'generate': lambda molecule, taper, freezecore: Moleculeclass(molecule, taper, freezecore).get_hartreefock_in_projector()
        },
        'qiskit_hf_and_energy': {
            'description': 'This is a simple Hartree-Fock Hamiltonian, multiplied by the Hartree-fock energy.',
            'generate': lambda molecule, taper, freezecore: Moleculeclass(molecule, taper, freezecore).get_hartreefock_energy()* Moleculeclass(molecule, taper, freezecore).get_hartreefock_in_projector()
        },

        'paper': {
            'description': 'Use Hartree-Fock from a specific paper (custom implementation)',
            'generate': lambda molecule, taper, freezecore:IBM_LiH_initial
        }
    },
    'final': {
        'qiskit': {
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