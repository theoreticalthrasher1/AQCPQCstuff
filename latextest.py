from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply some quantum gates
qc.h(0)  # Hadamard gate on qubit 0
qc.cx(0, 1)  # CNOT gate on qubits 0 and 1

# Generate LaTeX code
latex_code = qc.draw(output='latex')

# Print the LaTeX code
print(latex_code)
