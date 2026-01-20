import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

# --------------------------------------------------
# Tile decoding
# --------------------------------------------------
TILE_MAP = {
    "00": "Water",
    "01": "Sand",
    "10": "Grass",
    "11": "Jungle"
}

COLOR_MAP = {
    "Water": (0.2, 0.4, 1.0),
    "Sand": (0.9, 0.85, 0.6),
    "Grass": (0.2, 0.7, 0.2),
    "Jungle": (0.0, 0.4, 0.0)
}

# --------------------------------------------------
# Valid adjacency lookup
# --------------------------------------------------
VALID = {
    "Water": {"Water", "Sand"},
    "Sand": {"Water", "Sand", "Grass"},
    "Grass": {"Sand", "Grass", "Jungle"},
    "Jungle": {"Grass", "Jungle"}
}

# --------------------------------------------------
# Helper: compare two 2-qubit tiles
# --------------------------------------------------
def invalid_pair(tile_a, tile_b):
    return tile_b not in VALID[tile_a]

# --------------------------------------------------
# Oracle: mark ONLY invalid grids (flip phase)
# --------------------------------------------------
def grover_oracle(qc, data, flag):
    # We mark invalid configurations by flipping the flag qubit
    # when we encounter forbidden adjacencies
    
    for a, b in [(0,1),(0,2),(1,3),(2,3)]:
        for ta in TILE_MAP:
            for tb in TILE_MAP:
                if invalid_pair(TILE_MAP[ta], TILE_MAP[tb]):
                    idx = []
                    for i, bit in enumerate(ta):
                        if bit == "0":
                            qc.x(data[2*a+i])
                        idx.append(data[2*a+i])
                    for i, bit in enumerate(tb):
                        if bit == "0":
                            qc.x(data[2*b+i])
                        idx.append(data[2*b+i])

                    qc.mcx(idx, flag)

                    for i, bit in enumerate(ta):
                        if bit == "0":
                            qc.x(data[2*a+i])
                    for i, bit in enumerate(tb):
                        if bit == "0":
                            qc.x(data[2*b+i])

# --------------------------------------------------
# Diffusion
# --------------------------------------------------
def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)

# --------------------------------------------------
# Validate solution
# --------------------------------------------------
def is_valid_grid(bitstring):
    tiles = [bitstring[i:i+2] for i in range(0, 8, 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    
    # Check all adjacencies
    adjacencies = [(0,1), (0,2), (1,3), (2,3)]
    for a, b in adjacencies:
        if decoded[b] not in VALID[decoded[a]]:
            return False
    return True

# --------------------------------------------------
# Build Grover circuit with multiple iterations
# --------------------------------------------------
qc = QuantumCircuit(9, 8)
data = list(range(8))
flag = 8

# Initialize superposition
qc.h(data)
qc.x(flag)
qc.h(flag)

# Apply Grover iterations (optimal is ~sqrt(N/M) where N=256, M=valid solutions)
# Using 3-4 iterations for a 2x2 grid works well empirically
num_iterations = 3

for _ in range(num_iterations):
    grover_oracle(qc, data, flag)
    diffuser(qc, data)

# Measure
qc.measure(data, range(8))

# --------------------------------------------------
# Run
# --------------------------------------------------
backend = Aer.get_backend("qasm_simulator")
result = backend.run(qc, shots=8192).result()
counts = result.get_counts()

# --------------------------------------------------
# Filter for valid solutions only
# --------------------------------------------------
valid_counts = {k: v for k, v in counts.items() if is_valid_grid(k)}

if not valid_counts:
    print("No valid solutions found! Showing all results:")
    valid_counts = counts

# Find best valid solution
best = max(valid_counts, key=valid_counts.get)
tiles = [best[i:i+2] for i in range(0, 8, 2)]
decoded = [TILE_MAP[t] for t in tiles]

print(f"\nBest solution: {best}")
print(f"Decoded tiles: {decoded}")
print(f"Probability: {valid_counts[best]/sum(valid_counts.values())*100:.1f}%")
print(f"Is valid: {is_valid_grid(best)}")

# --------------------------------------------------
# Plot histogram (only valid solutions)
# --------------------------------------------------
plot_histogram(valid_counts)
plt.title("Valid WFC Configurations")
plt.show()

# --------------------------------------------------
# Plot WFC grid
# --------------------------------------------------
grid = np.zeros((2,2,3))
for i, tile in enumerate(decoded):
    r, c = divmod(i, 2)
    grid[r,c] = COLOR_MAP[tile]

plt.imshow(grid)
plt.axis("off")
plt.title(f"WFC Grid: {decoded}")
plt.show()