myaavqe=My_AAVQE(number_of_qubits,steps,layers,single_qubit_gates,entanglement_gates,entanglement,'qiskit_hf','qiskit',hfstate)
print(myaavqe.initial_hamiltonian)
H=myaavqe.initial_hamiltonian
# Assuming `H` is your operator
matrix = H.to_matrix()
# Find non-zero elements in the matrix
non_zero_indices = np.nonzero(matrix)  # Indices of non-zero elements
non_zero_values = matrix[non_zero_indices]  # Non-zero values
print(hfstate)
# Print the non-zero elements and their positions
print("Non-zero elements:")
for idx, value in zip(zip(*non_zero_indices), non_zero_values):
    print(f"Index: {idx}, Value: {value}")

# Assuming `H` is your operator and has been converted to a matrix form
matrix = H.to_matrix()

# Compute eigenvalues and eigenvectors using numpy.linalg.eig
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Find the index of the minimum eigenvalue
min_index = np.argmin(eigenvalues)

# Get the corresponding minimum eigenvector
min_eigenvector = eigenvectors[:, min_index]

# Normalize the eigenvector (optional but recommended)
min_eigenvector = min_eigenvector / np.linalg.norm(min_eigenvector)

# Print the minimum eigenvalue and corresponding eigenvector
print(f"Minimum eigenvalue: {eigenvalues[min_index]}")
print(f"Minimum eigenvector: {min_eigenvector}")
#myaavqe.initial_hamiltonian()
#myaavqe.alternative_run()
#print(myaavqe.draw_latex())