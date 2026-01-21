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

def calculate_iterations(num_tiles, num_valid):
    N = 4 ** num_tiles
    M = num_valid
    return max(1, int(np.floor(np.pi / (4 * np.arcsin(np.sqrt(M / N))))))

def check_invalid_pattern(qc, data, tile_a_idx, tile_b_idx, pattern_a, pattern_b, ancilla):
    """Check if a specific invalid pattern matches and mark ancilla"""
    controls = []
    
    # Build control list for tile A
    for i, bit in enumerate(pattern_a):
        if bit == "0":
            qc.x(data[2*tile_a_idx + i])
        controls.append(data[2*tile_a_idx + i])
    
    # Build control list for tile B
    for i, bit in enumerate(pattern_b):
        if bit == "0":
            qc.x(data[2*tile_b_idx + i])
        controls.append(data[2*tile_b_idx + i])
    
    # Mark ancilla if this invalid pattern is present
    qc.mcx(controls, ancilla)
    
    # Unflip the X gates
    for i, bit in enumerate(pattern_a):
        if bit == "0":
            qc.x(data[2*tile_a_idx + i])
    for i, bit in enumerate(pattern_b):
        if bit == "0":
            qc.x(data[2*tile_b_idx + i])

def grover_oracle_proper(qc, data, ancillas, output, adjacencies):
    """
    Proper oracle implementation:
    1. Check each adjacency for any invalid pattern -> set ancilla[i] = 1 if invalid
    2. Output = 1 if ALL ancillas are 0 (all adjacencies valid)
    3. Uncompute ancillas
    """
    num_ancillas = len(ancillas)
    
    # Step 1: For each adjacency, mark ancilla if ANY invalid pattern is present
    for adj_idx, (a, b) in enumerate(adjacencies):
        for ta in TILE_MAP:
            for tb in TILE_MAP:
                if invalid_pair(TILE_MAP[ta], TILE_MAP[tb]):
                    check_invalid_pattern(qc, data, a, b, ta, tb, ancillas[adj_idx])
    
    # Step 2: Flip all ancillas (now ancilla[i]=1 means valid)
    qc.x(ancillas)
    
    # Step 3: Output = 1 when ALL ancillas are 1 (all valid)
    qc.mcx(ancillas, output)
    
    # Step 4: Flip ancillas back
    qc.x(ancillas)
    
    # Step 5: Uncompute ancillas by repeating step 1 in reverse
    for adj_idx, (a, b) in enumerate(adjacencies):
        for ta in TILE_MAP:
            for tb in TILE_MAP:
                if invalid_pair(TILE_MAP[ta], TILE_MAP[tb]):
                    check_invalid_pattern(qc, data, a, b, ta, tb, ancillas[adj_idx])

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
    
    num_qubits = num_data_qubits + num_ancillas + 1  # No helpers needed
    
    print(f"Grid: {rows}×{cols}")
    print(f"Data qubits: {num_data_qubits}")
    print(f"Ancilla qubits: {num_ancillas}")
    print(f"Output qubit: 1")
    print(f"Total qubits: {num_qubits}")
    
    qc = QuantumCircuit(num_qubits, num_data_qubits)
    data = list(range(num_data_qubits))
    ancillas = list(range(num_data_qubits, num_data_qubits + num_ancillas))
    output = num_data_qubits + num_ancillas
    
    # Initialize: superposition on data qubits, output in |->
    qc.h(data)
    qc.x(output)
    qc.h(output)
    
    # Calculate optimal iterations
    num_valid = 54  # for 2x2 with your rules
    num_iterations = calculate_iterations(num_tiles, num_valid)
    print(f"Grover iterations: {num_iterations}")
    
    # Grover iterations
    for i in range(num_iterations):
        grover_oracle_proper(qc, data, ancillas, output, adjacencies)
        diffuser(qc, data)
    
    # Measure
    qc.measure(data, range(num_data_qubits))
    
    print("\nRunning simulation...")
    backend = Aer.get_backend("qasm_simulator")
    result = backend.run(qc, shots=8192).result()
    counts = result.get_counts()
    
    # Analyze results
    valid_counts = {k: v for k, v in counts.items() if is_valid_grid(k, adjacencies)}
    total_valid = sum(valid_counts.values())
    total_shots = sum(counts.values())
    
    print(f"\nResults:")
    print(f"Valid solutions: {total_valid}/{total_shots} ({100*total_valid/total_shots:.1f}%)")
    print(f"Unique valid configs: {len(valid_counts)}")
    
    if not valid_counts:
        print("WARNING: No valid solutions found!")
        valid_counts = counts
    
    # Get best solution
    best = max(valid_counts, key=valid_counts.get)
    tiles = [best[i:i+2] for i in range(0, len(best), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    
    print(f"\nMost common solution:")
    print(f"  Bitstring: {best}")
    print(f"  Decoded: {decoded}")
    print(f"  Count: {valid_counts[best]}")
    print(f"  Valid: {is_valid_grid(best, adjacencies)}")
    
    # Visualize
    grid = np.zeros((rows, cols, 3))
    for i, tile in enumerate(decoded):
        grid[i // cols, i % cols] = COLOR_MAP[tile]
    
    fig, ax = plt.subplots(figsize=(max(6, cols*2), max(6, rows*2)))
    ax.imshow(grid, interpolation='nearest')
    ax.axis("off")
    
    # Add labels
    for i, tile in enumerate(decoded):
        r, c = i // cols, i % cols
        ax.text(c, r, f"{i}\n{tile}", ha='center', va='center', 
                fontsize=max(8, 12-max(rows,cols)), color='white', weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # Add grid lines
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='black', linewidth=2)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color='black', linewidth=2)
    
    # Title with validity indicator
    title_color = 'green' if is_valid_grid(best, adjacencies) else 'red'
    ax.set_title(f"{rows}×{cols} WFC Grid (Grover)\nValid: {100*total_valid/total_shots:.1f}%", 
                 fontsize=14, color=title_color, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return qc, counts, decoded

if __name__ == "__main__":
    run_wfc_grover(GRID_ROWS, GRID_COLS)