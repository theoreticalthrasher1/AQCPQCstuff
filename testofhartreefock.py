
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_aer import Aer
driver = PySCFDriver(atom='H2', basis='sto-3g')
properties = driver.run()
num_spin_orbitals = driver.num_molecular_orbitals
num_particles = driver.num_alpha_electrons, driver.num_beta_electrons

init_state = HartreeFock(num_spin_orbitals, num_particles)
backend = Aer.get_backend('statevector_simulator')
job = execute(init_state, backend)
result = job.result()
statevector = result.result['statevector']
print(statevector)