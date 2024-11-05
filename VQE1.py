from preamble import *
from molecular import *
from qiskit_algorithms.utils import algorithm_globals
#seeds=[20, 21, 30, 33, 36, 42, 43, 55, 67,170 ]
seeds=67
algorithm_globals.random_seed= seeds
seed_transpiler = seeds
iterations = 125
shot = 6000
# Define parameters for each rotation gate

estimator = Estimator(
    backend_options = {
        'method': 'automatic',
        'device': 'CPU'
        # 'noise_model': noise_model
    },
    run_options = {
        'shots': shot,
        'seed': seeds,
    },
    transpile_options = {
        'seed_transpiler':seed_transpiler,
        'optimization_level': 3,

    },
    abelian_grouping = True
)
options = estimator.options
# Turn off all error mitigation and suppression
options.resilience_level = 2


theta_0 = Parameter('θ0')
theta_1 = Parameter('θ1')
theta_2 = Parameter('θ2')
theta_3 = Parameter('θ3')
phi_0 = Parameter('φ0')

phi_1 = Parameter('φ1')
phi_2 = Parameter('φ2')
phi_3 = Parameter('φ3')
lambda_0 = Parameter('λ0')
lambda_1 = Parameter('λ1')
lambda_2 = Parameter('λ2')
lambda_3 = Parameter('λ3')
# Initialize a quantum circuit with 6 qubits

qc = QuantumCircuit(tapered_op.num_qubits)

# Layer 1 with parameterized gates

qc.rx(theta_0, 0)
qc.ry(phi_0, 1)
qc.rz(lambda_0, 2)
qc.rx(theta_1, 3)
qc.cx(0, 1)
qc.cx(1, 2)
qc.crz(lambda_1, 2, 3)

# Layer 2 with parameterized gates
qc.rx(theta_2, 0)
qc.ry(phi_1, 1)
qc.rz(lambda_2, 2)
qc.rx(theta_3, 3)
qc.cx(2, 3)
qc.cx(0, 1)
qc.crz(lambda_3, 1, 2)

# Layer 3 with parameterized gates
qc.rx(phi_2, 0)
qc.ry(phi_3, 1)
qc.rz(lambda_0, 2)
qc.rx(theta_0, 3)
qc.cx(1, 2)
qc.cx(2, 3)
qc.crz(lambda_1, 0, 1)

# Repeat or add more layers as necessary
# For example, additional layers:
for _ in range(2):  # Replicate similar structure for more layers
    qc.rx(theta_0, 0)
    qc.ry(phi_0, 1)
    qc.rz(lambda_0, 2)
    qc.rx(theta_1, 3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.crz(lambda_1, 2, 3)

initial_state = HartreeFock(
    qmolecule.num_spatial_orbitals,
    qmolecule.num_particles,
    tapered_mapper,)


qc = initial_state.compose(qc)


# # Define a callback function to store the energy values
# energy_values_qc = []
# constant_energy_offset = -7.808849145498   # FreezeCoreTransformer extracted energy part
# nuclear_repulsion_energy= 1.0111666450700636

# def store_intermediate_result(eval_count, parameters, mean, std):
#     total_energy = mean + nuclear_repulsion_energy + constant_energy_offset
#     energy_values_qc.append(total_energy)


# vqe_qc = VQE(estimator, qc, optimizer=COBYLA(maxiter=iterations,tol=0.0001), callback=store_intermediate_result)
# vqe_qc.initial_point = [0.0] * qc.num_parameters
# start_time = time.time()
# calcqc = GroundStateEigensolver(tapered_mapper, vqe_qc)
# resqc = calcqc.solve(qmolecule)
# end_time = time.time()
# print(resqc)

# resultqc = resqc.computed_energies + resqc.extracted_transformer_energy + resqc.nuclear_repulsion_energy
# ref_value = ref_value.item() if isinstance(ref_value, np.ndarray) else ref_value
# resultqc = resultqc.item() if isinstance(resultqc, np.ndarray) else result

# error_rate_qc = abs(abs(ref_value - resultqc) / ref_value * 100)

# print("FreezeCoreTransformer extracted energy",resqc.extracted_transformer_energy)
# print("Ground state energy:", resqc.total_energies)
# print("Error rate: %f%%" % error_rate_qc)
# plt.plot(energy_values_qc)
# plt.axhline(y=ref_value, color='r',  linestyle='--', label='Reference Value')
# plt.xlabel('Iteration')
# plt.ylabel('Energy')
# plt.title(f'LiH Groundstate Energy (Custom QC) (Error rate: {error_rate_qc:.3f}%)')

# plt.legend()
# plt.show()

# Decompose the circuit to visualize its elementary gate structure
decomposed_qc = qc.decompose()
decomposed_qc.draw('mpl')
