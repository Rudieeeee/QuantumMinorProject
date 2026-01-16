from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --------------------------------------------------
# Helper: XNOR gate (checks equality)
# --------------------------------------------------
def xnor(qc, a, b, out):
    qc.cx(a, out)
    qc.cx(b, out)
    qc.x(out)

# --------------------------------------------------
# Build circuit
# --------------------------------------------------
qc = QuantumCircuit(9, 4)

# Data qubits
A, B, C, D = 0, 1, 2, 3

# Ancillas for constraints
AB, AC, BD, CD = 5, 6, 7, 8

# Oracle output qubit
FLAG = 4

# --------------------------------------------------
# 1️⃣ Superposition over all 2×2 grids
# --------------------------------------------------
qc.h([A, B, C, D])

# --------------------------------------------------
# 2️⃣ Oracle: check adjacency constraints
# Constraints:
# A == B
# A == C
# B == D
# C == D
# --------------------------------------------------
xnor(qc, A, B, AB)
xnor(qc, A, C, AC)
xnor(qc, B, D, BD)
xnor(qc, C, D, CD)

# AND all constraints → FLAG
qc.mcx([AB, AC, BD, CD], FLAG)

# Phase flip (Grover oracle)
qc.z(FLAG)

# --------------------------------------------------
# 3️⃣ Uncompute ancillas
# --------------------------------------------------
qc.mcx([AB, AC, BD, CD], FLAG)

xnor(qc, C, D, CD)
xnor(qc, B, D, BD)
xnor(qc, A, C, AC)
xnor(qc, A, B, AB)

# --------------------------------------------------
# 4️⃣ Diffusion operator (on grid qubits)
# --------------------------------------------------
qc.h([A, B, C, D])
qc.x([A, B, C, D])

qc.h(D)
qc.mcx([A, B, C], D)
qc.h(D)

qc.x([A, B, C, D])
qc.h([A, B, C, D])

# --------------------------------------------------
# 5️⃣ Measure grid
# --------------------------------------------------
qc.measure([A, B, C, D], [0, 1, 2, 3])

# --------------------------------------------------
# Run
# --------------------------------------------------
sim = AerSimulator()
result = sim.run(qc, shots=512).result()
counts = result.get_counts()

print(counts)
plot_histogram(counts)
plt.show()
