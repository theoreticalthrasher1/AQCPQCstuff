# %% [markdown]
# Install required package, we highly recommend participant to use qiskit platform, or at least participants can finish preprocessing at other platform and transfer the circuit to qiskit format, since our noise model is from IBM real machine backend and we restricted some algorithmic seeds which could be varied from different platform.

# %%
#!nvidia-smi

# %%
# !pip install qiskit
# !pip install qiskit-nature[pyscf] -U
# !pip install qiskit-ibm-runtime


# # %%
# !pip install qiskit_aer

# %%
#!pip install qiskit-aer-gpu

# %%
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit.circuit.library import TwoLocal
import numpy as np
import pylab
import qiskit.providers
from qiskit import pulse, QuantumCircuit, transpiler
import qiskit_nature
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE, AdaptVQE
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock, UCC
from qiskit.quantum_info import Pauli
qiskit_nature.settings.use_pauli_sum_op = False  # pylint: disable=undefined-variable
import matplotlib.pyplot as plt
from qiskit_algorithms import VQE
from qiskit_algorithms import optimizers
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper, TaperedQubitMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import ADAM , COBYLA, NELDER_MEAD,SPSA,L_BFGS_B, SLSQP
from qiskit_nature.second_q.algorithms import ExcitedStatesEigensolver,GroundStateEigensolver, QEOM, EvaluationRule
from qiskit_algorithms import NumPyEigensolver
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_aer import Aer , AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeTenerife, FakeVigoV2, FakeVigo, FakeAthens
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit.library import NLocal
from qiskit.circuit.library.n_local import EvolvedOperatorAnsatz
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info.analysis import Z2Symmetries
from functools import partial
from scipy.optimize import minimize
#from qiskit.quantum_info import I, X, Z, Y
from qiskit_algorithms.optimizers import (
 GradientDescent)
from qiskit_aer import Aer
from qiskit_aer.noise import \
 NoiseModel, depolarizing_error

import time
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper, TaperedQubitMapper
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# %%
from qiskit_algorithms.utils import algorithm_globals
#seeds=[20, 21, 30, 33, 36, 42, 43, 55, 67,170 ]
seeds=67
algorithm_globals.random_seed= seeds
seed_transpiler = seeds
iterations = 125
shot = 6000

# %%
