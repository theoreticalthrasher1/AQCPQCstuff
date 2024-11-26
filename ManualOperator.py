from qiskit.quantum_info import SparsePauliOp

import numpy as np

# Define Pauli strings and coefficients
pauli_strings_initial= ['IIII', 'ZIII', 'IZII', 'IIZI', 'IIIZ']
coefficients_initial = [0.568, -0.102, 0.245, 0.102, -0.245]

# Create the SparsePauliOp
pauli_op_initial = SparsePauliOp.from_list(list(zip(pauli_strings_initial, coefficients_initial)))

# Display the SparsePauliOp



part_1= [
    ("IIII",  0.567662), ("IZXX", -0.025425), ("ZXXX",  0.013812), ("XZYY", -0.011521), ("ZXZX", -0.008083),
    ("IIZI",  0.245088), ("IZYY",  0.025425), ("IXXX",  0.013812), ("XIYY",  0.011521), ("IXZX", -0.008083),
    ("ZIII", -0.245088), ("XXIZ",  0.025425), ("ZXYY", -0.013812), ("XZXX",  0.011521), ("ZXIX",  0.008083),
    ("IIZZ", -0.190085), ("YYIZ", -0.025425), ("IXYY", -0.013812), ("XIXX", -0.011521), ("IXIX",  0.008083),
    ("ZZII", -0.190085), ("IIXZ", -0.019768), ("XXZX", -0.013812), ("IIXX",  0.010474), ("ZXXZ", -0.006835),
    ("IZIZ", -0.107219), ("IIXI", -0.019768), ("YYZX",  0.013812), ("IIYY", -0.010474), ("IXXZ", -0.006835),
    ("IZII",  0.101581), ("XZII",  0.019768), ("XXIX",  0.013812), ("XXII",  0.010474), ("ZXXI", -0.006835),
    ("IIIZ", -0.101581), ("XIII", -0.019768), ("YYIX", -0.013812), ("YYII", -0.010474), ("IXXI", -0.006835),
    ("IZZI",  0.098833), ("XXZI", -0.018582), ("ZXZI", -0.012909), ("XZXI", -0.009880), ("XZZX", -0.006835),
    ("ZIIZ",  0.098833), ("YYZI",  0.018582), ("IXZI", -0.012909), ("XIXI",  0.009880), ("XZIX",  0.006835),
    ("ZIZI", -0.096556), ("ZIXX",  0.018582), ("ZIZX", -0.012909), ("XZXZ", -0.009880), ("XIZX",  0.006835),
    ("ZZZZ",  0.079438), ("ZIYY", -0.018582)]


# Separate Pauli strings and coefficients
pauli_strings = [term[0] for term in part_1]  # Extracts "IIII", "IZXX"
coefficients = [term[1] for term in part_1]  # Extracts 0.567662, -0.025425

# Create the SparsePauliOp
IBM_LiH = SparsePauliOp(pauli_strings, coefficients)

coefficients = [
    0.012909, 0.009880, -0.006835, -0.060240, 0.017442, -0.011861, 0.009298, -0.004511,
    0.060240, -0.017442, 0.011861, 0.009298, -0.004511, -0.053253, 0.017442, -0.011861,
    -0.009298, 0.004511, 0.053253, 0.017442, -0.011861, 0.009298, -0.004511, 0.033028,
    0.016652, -0.011521, -0.009044, -0.003631, -0.033028, 0.016652, 0.011521, 0.009044,
    0.003631, -0.033028, 0.016652, -0.011521, 0.009044, 0.003631, 0.033028, -0.016652,
    0.011521, 0.009044, -0.003631
]

pauli_strings = [
    "ZIIX", "XIXZ", "XIIX", "ZZZI", "IZZX", "XZZI", "ZZXI", "ZXZZ",
    "ZIZZ", "IZIX", "XIZI", "ZZXZ", "IXZZ", "IZZZ", "ZXIZ", "ZIXZ",
    "XZZZ", "ZZZX", "ZZIZ", "IXIZ", "ZIXI", "XIZZ", "ZZIX", "XXXX",
    "IZXZ", "XXXZ", "IIZX", "XXZZ", "YYXX", "IZXI", "YYXZ", "IIIX",
    "YYZZ", "XXYY", "XZIZ", "XXXI", "ZXII", "ZZYY", "YYYY", "XIIZ",
    "YYXI", "IXII", "ZZXX"
]

# Create the SparsePauliOp
IBM_LiH += SparsePauliOp(pauli_strings, coefficients) 
IBM_LiH_initial= pauli_op_initial 
# Output the SparsePauliOp
#print(IBM_LiH,IBM_LiH_initial)