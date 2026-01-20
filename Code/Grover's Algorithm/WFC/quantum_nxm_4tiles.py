from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# CONFIG
# ==================================================
ROWS = 2
COLS = 3
SHOTS = 40960
BITS_PER_TILE = 2  # 4 tiles

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
def tile_idx(r, c):
    return r * COLS + c

def bit_idx(tile, bit):
    return BITS_PER_TILE * tile + bit

# ==================================================
# Qubit counting
# ==================================================
N_TILES = ROWS * COLS
N_DATA = BITS_PER_TILE * N_TILES

# Each adjacency → 2 constraints (bit 0 + bit 1)
N_ADJ = (ROWS * (COLS - 1)) + ((ROWS - 1) * COLS)
N_CONSTRAINTS = 2 * N_ADJ

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
        t1 = tile_idx(r, c)
        t2 = tile_idx(r, c + 1)

        for bit in range(BITS_PER_TILE):
            a = bit_idx(t1, bit)
            b = bit_idx(t2, bit)
            out = CONSTRAINT_START + k
            xnor(qc, a, b, out)
            constraint_qubits.append(out)
            k += 1

# Vertical neighbors
for r in range(ROWS - 1):
    for c in range(COLS):
        t1 = tile_idx(r, c)
        t2 = tile_idx(r + 1, c)

        for bit in range(BITS_PER_TILE):
            a = bit_idx(t1, bit)
            b = bit_idx(t2, bit)
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

# Reverse vertical
for r in reversed(range(ROWS - 1)):
    for c in reversed(range(COLS)):
        t1 = tile_idx(r, c)
        t2 = tile_idx(r + 1, c)
        for bit in reversed(range(BITS_PER_TILE)):
            xnor(
                qc,
                bit_idx(t1, bit),
                bit_idx(t2, bit),
                CONSTRAINT_START + k
            )
            k -= 1

# Reverse horizontal
for r in reversed(range(ROWS)):
    for c in reversed(range(COLS - 1)):
        t1 = tile_idx(r, c)
        t2 = tile_idx(r, c + 1)
        for bit in reversed(range(BITS_PER_TILE)):
            xnor(
                qc,
                bit_idx(t1, bit),
                bit_idx(t2, bit),
                CONSTRAINT_START + k
            )
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
counts = sim.run(qc, shots=SHOTS).result().get_counts()

plot_histogram(counts)
plt.show()

# ==================================================
# Pick best state
# ==================================================
best_state = max(counts, key=counts.get)
bits = best_state[::-1]
print("Chosen state:", bits)

# ==================================================
# Render grid (decode 2-bit tiles)
# ==================================================
tile_colors = {
    "00": [0.2, 0.5, 1.0],   # water
    "01": [0.9, 0.85, 0.6],  # sand
    "10": [0.2, 0.8, 0.2],   # grass
    "11": [0.0, 0.5, 0.0],   # jungle
}

grid = np.zeros((ROWS, COLS, 3))
for r in range(ROWS):
    for c in range(COLS):
        t = tile_idx(r, c)
        tile_bits = bits[2*t:2*t+2]
        grid[r, c] = tile_colors[tile_bits]

plt.figure(figsize=(COLS, ROWS))
plt.imshow(grid)
plt.xticks([])
plt.yticks([])
plt.title("4-tile grid (only one tile globally)")
plt.show()
