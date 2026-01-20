import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer

GRID_ROWS = 2
GRID_COLS = 2

TILE_MAP = {"00": "Water", "01": "Sand", "10": "Grass", "11": "Jungle"}
COLOR_MAP = {"Water": (0.2, 0.4, 1.0), "Sand": (0.9, 0.85, 0.6), "Grass": (0.2, 0.7, 0.2), "Jungle": (0.0, 0.4, 0.0)}
VALID = {"Water": {"Water", "Sand"}, "Sand": {"Water", "Sand", "Grass"}, "Grass": {"Sand", "Grass", "Jungle"}, "Jungle": {"Grass", "Jungle"}}

def invalid_pair(tile_a, tile_b):
    return tile_b not in VALID[tile_a]

def get_adjacencies(rows, cols):
    adjacencies = []
    for row in range(rows):
        for col in range(cols - 1):
            adjacencies.append((row * cols + col, row * cols + col + 1))
    for row in range(rows - 1):
        for col in range(cols):
            adjacencies.append((row * cols + col, (row + 1) * cols + col))
    return adjacencies

def calculate_iterations(num_tiles):
    if num_tiles <= 4:
        return 4
    elif num_tiles <= 9:
        return 8
    else:
        return 12

def grover_oracle_with_helpers(qc, data, ancillas, helpers, output, adjacencies):
    # Step 1: For each adjacency, use helper to implement OR of all invalid patterns
    for adj_idx, (a, b) in enumerate(adjacencies):
        # Check each invalid pattern
        for ta in TILE_MAP:
            for tb in TILE_MAP:
                if invalid_pair(TILE_MAP[ta], TILE_MAP[tb]):
                    controls = []
                    for i, bit in enumerate(ta):
                        if bit == "0":
                            qc.x(data[2*a+i])
                        controls.append(data[2*a+i])
                    for i, bit in enumerate(tb):
                        if bit == "0":
                            qc.x(data[2*b+i])
                        controls.append(data[2*b+i])
                    
                    # Set helper to 1 if this pattern matches
                    qc.mcx(controls, helpers[adj_idx])
                    
                    for i, bit in enumerate(ta):
                        if bit == "0":
                            qc.x(data[2*a+i])
                    for i, bit in enumerate(tb):
                        if bit == "0":
                            qc.x(data[2*b+i])
        
        # Now ancilla = NOT(helper): ancilla is 1 if adjacency is VALID (helper is 0)
        qc.x(helpers[adj_idx])
        qc.cx(helpers[adj_idx], ancillas[adj_idx])
        qc.x(helpers[adj_idx])
    
    # Step 2: Output = 1 when ALL ancillas are 1 (all adjacencies valid)
    qc.mcx(ancillas, output)
    
    # Step 3: Uncompute ancillas
    for adj_idx, (a, b) in enumerate(adjacencies):
        qc.x(helpers[adj_idx])
        qc.cx(helpers[adj_idx], ancillas[adj_idx])
        qc.x(helpers[adj_idx])
    
    # Step 4: Uncompute helpers
    for adj_idx, (a, b) in enumerate(adjacencies):
        for ta in TILE_MAP:
            for tb in TILE_MAP:
                if invalid_pair(TILE_MAP[ta], TILE_MAP[tb]):
                    controls = []
                    for i, bit in enumerate(ta):
                        if bit == "0":
                            qc.x(data[2*a+i])
                        controls.append(data[2*a+i])
                    for i, bit in enumerate(tb):
                        if bit == "0":
                            qc.x(data[2*b+i])
                        controls.append(data[2*b+i])
                    
                    qc.mcx(controls, helpers[adj_idx])
                    
                    for i, bit in enumerate(ta):
                        if bit == "0":
                            qc.x(data[2*a+i])
                    for i, bit in enumerate(tb):
                        if bit == "0":
                            qc.x(data[2*b+i])

def diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)

def is_valid_grid(bitstring, adjacencies):
    tiles = [bitstring[i:i+2] for i in range(0, len(bitstring), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    for a, b in adjacencies:
        if decoded[b] not in VALID[decoded[a]]:
            return False
    return True

def run_wfc_grover(rows, cols):
    num_tiles = rows * cols
    num_data_qubits = num_tiles * 2
    adjacencies = get_adjacencies(rows, cols)
    num_ancillas = len(adjacencies)
    num_helpers = len(adjacencies)  # One helper per adjacency for OR logic
    
    num_qubits = num_data_qubits + num_ancillas + num_helpers + 1
    
    print(f"Grid: {rows}×{cols}")
    print(f"Data: {num_data_qubits}, Ancillas: {num_ancillas}, Helpers: {num_helpers}, Output: 1")
    print(f"Total qubits: {num_qubits}")
    
    qc = QuantumCircuit(num_qubits, num_data_qubits)
    data = list(range(num_data_qubits))
    ancillas = list(range(num_data_qubits, num_data_qubits + num_ancillas))
    helpers = list(range(num_data_qubits + num_ancillas, num_data_qubits + num_ancillas + num_helpers))
    output = num_data_qubits + num_ancillas + num_helpers
    
    qc.h(data)
    qc.x(output)
    qc.h(output)
    
    num_iterations = calculate_iterations(num_tiles)
    print(f"Iterations: {num_iterations}")
    
    for i in range(num_iterations):
        grover_oracle_with_helpers(qc, data, ancillas, helpers, output, adjacencies)
        diffuser(qc, data)
    
    qc.measure(data, range(num_data_qubits))
    
    print("Running simulation...")
    backend = Aer.get_backend("qasm_simulator")
    result = backend.run(qc, shots=8192).result()
    counts = result.get_counts()
    
    valid_counts = {k: v for k, v in counts.items() if is_valid_grid(k, adjacencies)}
    total_valid = sum(valid_counts.values())
    total_shots = sum(counts.values())
    
    print(f"\nValid: {total_valid}/{total_shots} ({100*total_valid/total_shots:.1f}%)")
    print(f"Unique valid configs: {len(valid_counts)}")
    
    if not valid_counts:
        print("WARNING: No valid solutions found!")
        valid_counts = counts
    
    best = max(valid_counts, key=valid_counts.get)
    tiles = [best[i:i+2] for i in range(0, len(best), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    
    print(f"Best solution: {decoded}")
    print(f"Is valid: {is_valid_grid(best, adjacencies)}")
    
    grid = np.zeros((rows, cols, 3))
    for i, tile in enumerate(decoded):
        grid[i // cols, i % cols] = COLOR_MAP[tile]
    
    fig, ax = plt.subplots(figsize=(max(6, cols*2), max(6, rows*2)))
    ax.imshow(grid, interpolation='nearest')
    ax.axis("off")
    
    for i, tile in enumerate(decoded):
        r, c = i // cols, i % cols
        ax.text(c, r, f"{i}\n{tile}", ha='center', va='center', fontsize=max(8, 12-max(rows,cols)), 
                color='white', weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='black', linewidth=2)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color='black', linewidth=2)
    
    title_color = 'green' if is_valid_grid(best, adjacencies) else 'red'
    if rows <= 3 and cols <= 3:
        grid_str = '\n'.join([str(decoded[row*cols:(row+1)*cols]) for row in range(rows)])
        ax.set_title(f"{rows}×{cols} WFC Grid\n{grid_str}", fontsize=11, color=title_color, weight='bold')
    else:
        ax.set_title(f"{rows}×{cols} WFC Grid", fontsize=14, color=title_color, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return qc, counts, decoded

if __name__ == "__main__":
    run_wfc_grover(GRID_ROWS, GRID_COLS)