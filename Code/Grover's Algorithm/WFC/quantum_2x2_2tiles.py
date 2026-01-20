from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

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

# Ancillas
AB, AC, BD, CD = 5, 6, 7, 8

# Oracle output
FLAG = 4

# --------------------------------------------------
# 1️⃣ Superposition
# --------------------------------------------------
qc.h([A, B, C, D])

# --------------------------------------------------
# 2️⃣ Oracle constraints
# --------------------------------------------------
xnor(qc, A, B, AB)
xnor(qc, A, C, AC)
xnor(qc, B, D, BD)
xnor(qc, C, D, CD)

qc.mcx([AB, AC, BD, CD], FLAG)
qc.z(FLAG)

# --------------------------------------------------
# 3️⃣ Uncompute
# --------------------------------------------------
qc.mcx([AB, AC, BD, CD], FLAG)

xnor(qc, C, D, CD)
xnor(qc, B, D, BD)
xnor(qc, A, C, AC)
xnor(qc, A, B, AB)

# --------------------------------------------------
# 4️⃣ Diffusion
# --------------------------------------------------
qc.h([A, B, C, D])
qc.x([A, B, C, D])

qc.h(D)
qc.mcx([A, B, C], D)
qc.h(D)

qc.x([A, B, C, D])
qc.h([A, B, C, D])

# --------------------------------------------------
# 5️⃣ Measure
# --------------------------------------------------
qc.measure([A, B, C, D], [0, 1, 2, 3])

# --------------------------------------------------
# Run
# --------------------------------------------------
sim = AerSimulator()
result = sim.run(qc, shots=512).result()
counts = result.get_counts()

print("Measurement counts:")
print(counts)

plot_histogram(counts)
plt.show()

# --------------------------------------------------
# Pick highest probability state
# --------------------------------------------------
best_state = max(counts, key=counts.get)
print("Chosen state:", best_state)

# Qiskit order: D C B A → reverse
bits = best_state[::-1]

# --------------------------------------------------
# Render grid image
# --------------------------------------------------
color_map = {
    "0": [0.2, 0.5, 1.0],  # water (blue)
    "1": [0.2, 0.8, 0.2],  # grass (green)
}

grid = np.array([
    [color_map[bits[0]], color_map[bits[1]]],
    [color_map[bits[2]], color_map[bits[3]]],
])

plt.figure(figsize=(4, 4))
plt.imshow(grid)
plt.xticks([])
plt.yticks([])
plt.title("Generated 2×2 Tile Grid\n(0 = water, 1 = grass)")
plt.show()
