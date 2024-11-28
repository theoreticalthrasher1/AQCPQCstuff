from qiskit.quantum_info import SparsePauliOp

def get_string_from_dict(pauli_dict: dict) -> str:
    """Extracts the Pauli string from a dictionary.

    Args:
        pauli_dict: A dictionary containing a single key (Pauli string).

    Returns:
        The Pauli string as a standard Python string.
    """
    pauli_str_np = list(pauli_dict.keys())[0]  # Get the key (NumPy string)
    pauli_str = str(pauli_str_np)            # Convert to standard Python string
    return pauli_str

# def create_z_operator_from_binary(binary_list: list) -> SparsePauliOp:
#     """Creates a SparsePauliOp with Z gates based on a binary list.

#     Args:
#         binary_list: A list of 0s and 1s indicating qubit targets for Z gates.

#     Returns:
#         A SparsePauliOp representing the Z gate operations.
#     """
#     num_qubits = len(binary_list)
#     pauli_string = ['I'] * num_qubits  # Initialize with identities

#     for index, value in enumerate(binary_list):
#         if value == 1:
#             pauli_string[index] = 'Z'  # Place 'Z' where needed

#     return SparsePauliOp([''.join(pauli_string)], [1])  
def create_projector_from_binary_string(binary_string: str) -> SparsePauliOp:
    """Creates a SparsePauliOp with Z gates based on a binary string.

    Args:
        binary_string: A string of 0s and 1s indicating qubit targets for Z gates.

    Returns:
        A SparsePauliOp representing the Z gate operations.
    """
    
    # List to store Pauli operator terms
    pauli_ops = []
    coeffs = []
    
    for bit in binary_string:
        if bit == '0':
            # For 0, use (1/2)*(I + Z)
            pauli_ops.append('I')
            coeffs.append(0.5)
            pauli_ops.append('Z')
            coeffs.append(0.5)
        elif bit == '1':
            # For 1, use (1/2)*(I - Z)
            pauli_ops.append('I')
            coeffs.append(0.5)
            pauli_ops.append('Z')
            coeffs.append(-0.5)
    
    # Convert the list of operators and coefficients to a SparsePauliOp
    return ['I']*len(binary_string)- SparsePauliOp(pauli_ops, coeffs)

test=create_projector_from_binary_string('0')
print(test)
