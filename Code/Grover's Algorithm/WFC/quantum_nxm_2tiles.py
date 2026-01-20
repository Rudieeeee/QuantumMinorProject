from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# CONFIG
# ==================================================
ROWS = 2
COLS = 2
SHOTS = 4096

# ==================================================
# Helper: XNOR
# ==================================================
def xnor(qc, a, b, out):
    qc.cx(a, out)
    qc.cx(b, out)
    qc.x(out)

# ==================================================
# Index helpers
# ==================================================
def idx(r, c):
    return r * COLS + c

# ==================================================
# Qubit counting
# ==================================================
N_DATA = ROWS * COLS

# Horizontal + vertical adjacency constraints
N_CONSTRAINTS = (ROWS * (COLS - 1)) + ((ROWS - 1) * COLS)

DATA_START = 0
FLAG = N_DATA
CONSTRAINT_START = FLAG + 1

TOTAL_QUBITS = N_DATA + 1 + N_CONSTRAINTS

qc = QuantumCircuit(TOTAL_QUBITS, N_DATA)

# ==================================================
# 1️⃣ Superposition
# ==================================================
qc.h(range(N_DATA))

# ==================================================
# 2️⃣ Oracle: adjacency constraints
# ==================================================
constraint_qubits = []
k = 0

# Horizontal neighbors
for r in range(ROWS):
    for c in range(COLS - 1):
        a = idx(r, c)
        b = idx(r, c + 1)
        out = CONSTRAINT_START + k
        xnor(qc, a, b, out)
        constraint_qubits.append(out)
        k += 1

# Vertical neighbors
for r in range(ROWS - 1):
    for c in range(COLS):
        a = idx(r, c)
        b = idx(r + 1, c)
        out = CONSTRAINT_START + k
        xnor(qc, a, b, out)
        constraint_qubits.append(out)
        k += 1

# AND all constraints → FLAG
qc.mcx(constraint_qubits, FLAG)
qc.z(FLAG)

# ==================================================
# 3️⃣ Uncompute constraints
# ==================================================
qc.mcx(constraint_qubits, FLAG)

k -= 1
for r in reversed(range(ROWS - 1)):
    for c in reversed(range(COLS)):
        xnor(qc, idx(r, c), idx(r + 1, c), CONSTRAINT_START + k)
        k -= 1

for r in reversed(range(ROWS)):
    for c in reversed(range(COLS - 1)):
        xnor(qc, idx(r, c), idx(r, c + 1), CONSTRAINT_START + k)
        k -= 1

# ==================================================
# 4️⃣ Diffusion operator
# ==================================================
qc.h(range(N_DATA))
qc.x(range(N_DATA))

qc.h(N_DATA - 1)
qc.mcx(list(range(N_DATA - 1)), N_DATA - 1)
qc.h(N_DATA - 1)

qc.x(range(N_DATA))
qc.h(range(N_DATA))

# ==================================================
# 5️⃣ Measurement
# ==================================================
qc.measure(range(N_DATA), range(N_DATA))

# ==================================================
# Run
# ==================================================
sim = AerSimulator()
result = sim.run(qc, shots=SHOTS).result()
counts = result.get_counts()

plot_histogram(counts)
plt.show()

# ==================================================
# Pick best state
# ==================================================
best_state = max(counts, key=counts.get)
bits = best_state[::-1]

print("Chosen state:", bits)

# ==================================================
# Render grid
# ==================================================
color_map = {
    "0": [0.2, 0.5, 1.0],  # water
    "1": [0.2, 0.8, 0.2],  # grass
}

grid = np.zeros((ROWS, COLS, 3))
for r in range(ROWS):
    for c in range(COLS):
        grid[r, c] = color_map[bits[idx(r, c)]]

plt.figure(figsize=(COLS, ROWS))
plt.imshow(grid)
plt.xticks([])
plt.yticks([])
plt.title(f"Generated {ROWS}×{COLS} Tile Grid\n(0 = water, 1 = grass)")
plt.show()
