from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# XNOR helper
# --------------------------------------------------
def xnor(qc, a, b, out):
    qc.cx(a, out)
    qc.cx(b, out)
    qc.x(out)

# --------------------------------------------------
# Circuit
# --------------------------------------------------
qc = QuantumCircuit(22, 9)

# --------------------------------------------------
# Data qubits (3x3 grid)
# --------------------------------------------------
A, B, C = 0, 1, 2
D, E, F = 3, 4, 5
G, H, I = 6, 7, 8

# Oracle flag
FLAG = 9

# Constraint ancillas
AB, BC = 10, 11
DE, EF = 12, 13
GH, HI = 14, 15

AD, DG = 16, 17
BE, EH = 18, 19
CF, FI = 20, 21

# --------------------------------------------------
# 1️⃣ Superposition
# --------------------------------------------------
qc.h(range(9))

# --------------------------------------------------
# 2️⃣ Oracle: adjacency constraints
# --------------------------------------------------

# Horizontal
xnor(qc, A, B, AB)
xnor(qc, B, C, BC)
xnor(qc, D, E, DE)
xnor(qc, E, F, EF)
xnor(qc, G, H, GH)
xnor(qc, H, I, HI)

# Vertical
xnor(qc, A, D, AD)
xnor(qc, D, G, DG)
xnor(qc, B, E, BE)
xnor(qc, E, H, EH)
xnor(qc, C, F, CF)
xnor(qc, F, I, FI)

# AND all constraints → FLAG
qc.mcx(
    [AB, BC, DE, EF, GH, HI, AD, DG, BE, EH, CF, FI],
    FLAG
)

# Phase flip
qc.z(FLAG)

# --------------------------------------------------
# 3️⃣ Uncompute constraints
# --------------------------------------------------
qc.mcx(
    [AB, BC, DE, EF, GH, HI, AD, DG, BE, EH, CF, FI],
    FLAG
)

# Vertical (reverse)
xnor(qc, F, I, FI)
xnor(qc, C, F, CF)
xnor(qc, E, H, EH)
xnor(qc, B, E, BE)
xnor(qc, D, G, DG)
xnor(qc, A, D, AD)

# Horizontal (reverse)
xnor(qc, H, I, HI)
xnor(qc, G, H, GH)
xnor(qc, E, F, EF)
xnor(qc, D, E, DE)
xnor(qc, B, C, BC)
xnor(qc, A, B, AB)

# --------------------------------------------------
# 4️⃣ Diffusion operator
# --------------------------------------------------
qc.h(range(9))
qc.x(range(9))

qc.h(I)
qc.mcx([A, B, C, D, E, F, G, H], I)
qc.h(I)

qc.x(range(9))
qc.h(range(9))

# --------------------------------------------------
# 5️⃣ Measurement
# --------------------------------------------------
qc.measure(range(9), range(9))

# --------------------------------------------------
# Run
# --------------------------------------------------
sim = AerSimulator()
result = sim.run(qc, shots=20000).result()
counts = result.get_counts()

print("Measurement counts:")
print(counts)

plot_histogram(counts)
plt.show()

# --------------------------------------------------
# Pick highest probability result
# --------------------------------------------------
best_state = max(counts, key=counts.get)
print("Chosen state:", best_state)

# Qiskit order: I H G F E D C B A → reverse
bits = best_state[::-1]

# --------------------------------------------------
# Render 3x3 grid image
# --------------------------------------------------
color_map = {
    "0": [0.2, 0.5, 1.0],  # water
    "1": [0.2, 0.8, 0.2],  # grass
}

grid = np.array([
    [color_map[bits[0]], color_map[bits[1]], color_map[bits[2]]],
    [color_map[bits[3]], color_map[bits[4]], color_map[bits[5]]],
    [color_map[bits[6]], color_map[bits[7]], color_map[bits[8]]],
])

plt.figure(figsize=(5, 5))
plt.imshow(grid)
plt.xticks([])
plt.yticks([])
plt.title("Generated 3×3 Tile Grid\n(0 = water, 1 = grass)")
plt.show()
