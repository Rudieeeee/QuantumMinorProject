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
# Generate adjacency pairs for 3x3 grid
# Grid layout:
# 0 1 2
# 3 4 5
# 6 7 8
# --------------------------------------------------
def get_adjacencies_3x3():
    adjacencies = []
    # Horizontal adjacencies
    for row in range(3):
        for col in range(2):
            left = row * 3 + col
            right = row * 3 + col + 1
            adjacencies.append((left, right))
    
    # Vertical adjacencies
    for row in range(2):
        for col in range(3):
            top = row * 3 + col
            bottom = (row + 1) * 3 + col
            adjacencies.append((top, bottom))
    
    return adjacencies

# --------------------------------------------------
# Oracle: mark ONLY invalid grids (flip phase)
# --------------------------------------------------
def grover_oracle(qc, data, flag, adjacencies):
    # We mark invalid configurations by flipping the flag qubit
    # when we encounter forbidden adjacencies
    
    for a, b in adjacencies:
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
def is_valid_grid(bitstring, adjacencies):
    tiles = [bitstring[i:i+2] for i in range(0, len(bitstring), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    
    # Check all adjacencies
    for a, b in adjacencies:
        if decoded[b] not in VALID[decoded[a]]:
            return False
    return True

# --------------------------------------------------
# Build Grover circuit with multiple iterations
# --------------------------------------------------
# 3x3 grid = 9 tiles, each tile needs 2 qubits = 18 data qubits + 1 flag qubit
num_tiles = 9
num_data_qubits = num_tiles * 2
num_qubits = num_data_qubits + 1

qc = QuantumCircuit(num_qubits, num_data_qubits)
data = list(range(num_data_qubits))
flag = num_data_qubits

adjacencies = get_adjacencies_3x3()
print(f"3x3 Grid adjacencies: {adjacencies}")
print(f"Total adjacencies: {len(adjacencies)}")
print(f"Total qubits: {num_qubits} (18 data + 1 flag)")

# Initialize superposition (all 2^18 = 262,144 possible grids)
qc.h(data)
qc.x(flag)
qc.h(flag)

# Apply Grover iterations
# For 3x3, the search space is MUCH larger (2^18 vs 2^8)
# So we need more iterations: optimal ≈ π/4 * sqrt(2^18 / num_valid)
# Using 8-12 iterations works well empirically
num_iterations = 10

print(f"Running {num_iterations} Grover iterations...")

for iteration in range(num_iterations):
    grover_oracle(qc, data, flag, adjacencies)
    diffuser(qc, data)
    if (iteration + 1) % 2 == 0:
        print(f"  Completed iteration {iteration + 1}/{num_iterations}")

# Measure
qc.measure(data, range(num_data_qubits))

# --------------------------------------------------
# Run
# --------------------------------------------------
print("Running quantum simulation...")
backend = Aer.get_backend("qasm_simulator")
result = backend.run(qc, shots=8192).result()
counts = result.get_counts()

# --------------------------------------------------
# Filter for valid solutions only
# --------------------------------------------------
valid_counts = {k: v for k, v in counts.items() if is_valid_grid(k, adjacencies)}
invalid_counts = {k: v for k, v in counts.items() if not is_valid_grid(k, adjacencies)}

total_valid = sum(valid_counts.values())
total_invalid = sum(invalid_counts.values())

print(f"\nResults:")
print(f"Valid grids: {total_valid}/{total_valid+total_invalid} ({100*total_valid/(total_valid+total_invalid):.1f}%)")
print(f"Invalid grids: {total_invalid}/{total_valid+total_invalid} ({100*total_invalid/(total_valid+total_invalid):.1f}%)")

if not valid_counts:
    print("\nWarning: No valid solutions found! Showing all results:")
    valid_counts = counts

# Find best valid solution
best = max(valid_counts, key=valid_counts.get)
tiles = [best[i:i+2] for i in range(0, len(best), 2)]
decoded = [TILE_MAP[t] for t in tiles]

print(f"\nBest solution: {best}")
print(f"Decoded tiles: {decoded}")
print(f"Count: {valid_counts[best]} shots")
print(f"Probability: {valid_counts[best]/sum(valid_counts.values())*100:.1f}%")
print(f"Is valid: {is_valid_grid(best, adjacencies)}")

# Verify adjacencies
print("\nAdjacency validation:")
for a, b in adjacencies:
    status = "✓" if decoded[b] in VALID[decoded[a]] else "✗"
    print(f"  {status} Position {a}({decoded[a]}) → Position {b}({decoded[b]})")

# --------------------------------------------------
# Plot histogram (top 20 valid solutions)
# --------------------------------------------------
sorted_valid = dict(sorted(valid_counts.items(), key=lambda x: x[1], reverse=True)[:20])
plot_histogram(sorted_valid)
plt.title(f"Top 20 Valid WFC Configurations (3x3)")
plt.show()

# --------------------------------------------------
# Plot WFC grid (3x3)
# --------------------------------------------------
grid = np.zeros((3, 3, 3))
for i, tile in enumerate(decoded):
    r = i // 3
    c = i % 3
    grid[r, c] = COLOR_MAP[tile]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(grid, interpolation='nearest')
ax.axis("off")

# Add tile labels
for i, tile in enumerate(decoded):
    r = i // 3
    c = i % 3
    ax.text(c, r, f"{i}\n{tile}", ha='center', va='center', 
            fontsize=10, color='white', weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

title_color = 'green' if is_valid_grid(best, adjacencies) else 'red'
ax.set_title(f"3x3 WFC Grid\n{decoded[:3]}\n{decoded[3:6]}\n{decoded[6:9]}", 
             fontsize=11, color=title_color, weight='bold')
plt.tight_layout()
plt.show()