from qiskit import QuantumCircuit

# Create a quantum circuit
qc = QuantumCircuit(2, 2)

# Apply some gates
qc.h(0)  # Hadamard on qubit 0
qc.cx(0, 1)  # CNOT on qubits 0 and 1

# Generate LaTeX code for the circuit (not an image)
latex_code = qc.draw(output='latex')

# Print the LaTeX code
print(latex_code)
