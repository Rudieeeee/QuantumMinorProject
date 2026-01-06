# -------------------------
# Built-in modules
# -------------------------
import math
import matplotlib.pyplot as plt

# -------------------------
# Qiskit imports
# -------------------------
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit.visualization import plot_distribution

# -------------------------
# Aer simulator (modern)
# -------------------------
from qiskit_aer import AerSimulator


# --------------------------------------------------
# Oracle definition (from official Qiskit tutorial)
# --------------------------------------------------
def grover_oracle(marked_states):
    if not isinstance(marked_states, list):
        marked_states = [marked_states]

    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)

    for target in marked_states:
        # Reverse bitstring (Qiskit convention)
        rev_target = target[::-1]

        # Indices where target bit is 0
        zero_inds = [i for i in range(num_qubits) if rev_target[i] == "0"]

        # Apply X for open controls
        if zero_inds:
            qc.x(zero_inds)

        # Multi-controlled Z
        qc.compose(
            MCMTGate(ZGate(), num_qubits - 1, 1),
            inplace=True
        )

        # Undo X
        if zero_inds:
            qc.x(zero_inds)

    return qc


# --------------------------------------------------
# Define Grover problem
# --------------------------------------------------
marked_states = ["011", "100"]  # marked states

oracle = grover_oracle(marked_states)
grover_op = grover_operator(oracle)

# Optimal number of Grover iterations
optimal_num_iterations = math.floor(
    math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2**grover_op.num_qubits)))
)

# --------------------------------------------------
# Build full Grover circuit
# --------------------------------------------------
qc = QuantumCircuit(grover_op.num_qubits)

# Initialize uniform superposition
qc.h(range(grover_op.num_qubits))

# Apply Grover operator
qc.compose(grover_op.power(optimal_num_iterations), inplace=True)

# Measure
qc.measure_all()

print("Original (high-level) circuit:")
print(qc.draw())

# --------------------------------------------------
# Run on AerSimulator
# --------------------------------------------------
backend = AerSimulator()

# 🔴 CRITICAL STEP: FORCE FULL DECOMPOSITION
qc_transpiled = transpile(
    qc,
    backend=backend,
    basis_gates=["u3", "cx"],
    optimization_level=3
)

print("\nTranspiled (hardware-level) circuit:")
print(qc_transpiled.draw())

# Execute
job = backend.run(qc_transpiled, shots=10_000)
result = job.result()

counts = result.get_counts()
print("\nMeasurement distribution:")
print(counts)

# Plot result
plot_distribution(counts)
plt.show()
