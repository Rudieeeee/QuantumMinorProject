from itertools import chain
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit_aer import Aer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt

GRID_ROW_COUNT = 2
GRID_COL_COUNT = 2
GRID_TILE_COUNT = GRID_ROW_COUNT * GRID_COL_COUNT
GRID_EDGES = []
for row in range(GRID_ROW_COUNT):
    for col in range(GRID_COL_COUNT - 1):
        GRID_EDGES.append((GRID_COL_COUNT * row + col, GRID_COL_COUNT * row + col + 1))
for row in range(GRID_ROW_COUNT - 1):
    for col in range(GRID_COL_COUNT):
        GRID_EDGES.append((GRID_COL_COUNT * row + col, GRID_COL_COUNT * (row + 1) + col))
        
DATA_QUBIT_COUNT = GRID_TILE_COUNT * 2
HELPER_QUBIT_COUNT = 2  # Just 2 helpers, reused for all edges!
RESULT_QUBIT_COUNT = len(GRID_EDGES)  # One result qubit per edge
TOTAL_QUBIT_COUNT = DATA_QUBIT_COUNT + HELPER_QUBIT_COUNT + RESULT_QUBIT_COUNT
    
DATA_QUBITS = list(range(DATA_QUBIT_COUNT))
HELPER_QUBITS = list(range(DATA_QUBIT_COUNT, DATA_QUBIT_COUNT + HELPER_QUBIT_COUNT))
RESULT_QUBITS = list(range(DATA_QUBIT_COUNT + HELPER_QUBIT_COUNT, TOTAL_QUBIT_COUNT))

TILE_MAP = {"00": "Water", "01": "Sand", "10": "Grass", "11": "Jungle"}
COLOR_MAP = {"Water": (0.2, 0.4, 1.0), "Sand": (0.9, 0.85, 0.6), "Grass": (0.2, 0.7, 0.2), "Jungle": (0.0, 0.4, 0.0)}
VALID = {"Water": {"Water", "Sand"}, "Sand": {"Water", "Sand", "Grass"}, "Grass": {"Sand", "Grass", "Jungle"}, "Jungle": {"Grass", "Jungle"}}

def is_valid_grid(bitstring):
    tiles = [bitstring[i:i+2] for i in range(0, len(bitstring), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    for a, b in GRID_EDGES:
        if decoded[b] not in VALID[decoded[a]]:
            return False
    return True

def checkerboard_x(qc):
    for row in range(GRID_ROW_COUNT):
        for col in range(GRID_COL_COUNT):
            if (row + col) % 2 == 1:
                i = GRID_COL_COUNT * row + col
                data = DATA_QUBITS[i*2:i*2+2]
                qc.x(data)

def check_single_edge(qc, edge_idx):
    """
    Check a single edge using the 2 helper qubits, store result in result_qubit
    Then uncompute the helpers (reset them to |0>)
    """
    a, b = GRID_EDGES[edge_idx]
    helpers = HELPER_QUBITS  # Always use the same 2 helpers
    result = RESULT_QUBITS[edge_idx]  # But store in different result qubit per edge
    data_a = DATA_QUBITS[a*2:a*2+2]
    data_b = DATA_QUBITS[b*2:b*2+2]
    
    # Phase 1: Mark helpers based on constraint patterns
    qc.mcx([data_a[1], data_b[1], data_a[0]], helpers[0])
    qc.mcx([data_a[1], data_b[1], data_b[0]], helpers[1])
    
    # Phase 2: Check with flipped encodings
    qc.x(data_a)
    qc.x(data_b)
    qc.mcx([data_a[1], data_b[1], data_a[0]], helpers[0])
    qc.mcx([data_a[1], data_b[1], data_b[0]], helpers[1])
    qc.x(data_a)
    qc.x(data_b)
    
    # Copy helpers to result: result = helpers[0] OR helpers[1]
    # If either helper is 1, the edge is invalid
    # We want result = 1 if edge is VALID (both helpers = 0)
    qc.x(helpers)  # Flip: now helpers = 1 means no invalid pattern
    qc.ccx(helpers[0], helpers[1], result)  # result = 1 if both helpers = 1
    qc.x(helpers)  # Flip back
    
    # Uncompute helpers (reverse the constraint checks)
    qc.x(data_a)
    qc.x(data_b)
    qc.mcx([data_a[1], data_b[1], data_a[0]], helpers[0])
    qc.mcx([data_a[1], data_b[1], data_b[0]], helpers[1])
    qc.x(data_a)
    qc.x(data_b)
    
    qc.mcx([data_a[1], data_b[1], data_a[0]], helpers[0])
    qc.mcx([data_a[1], data_b[1], data_b[0]], helpers[1])
    
    # Now helpers are back to |00> and can be reused!

def constraints_with_reuse():
    """
    Check all edges sequentially, reusing the same 2 helper qubits
    Results are stored in separate result qubits
    """
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT)
    
    # Check each edge one by one, reusing helpers
    for i in range(len(GRID_EDGES)):
        check_single_edge(qc, i)
    
    return qc

def grover_oracle():
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT)
    
    # Step 1: Compute all result qubits using reused helpers
    qc.compose(constraints_with_reuse(), inplace=True)
    
    # Step 2: Apply multi-controlled Z when ALL result qubits = 1 (all edges valid)
    qc.compose(MCMTGate(ZGate(), len(RESULT_QUBITS), len(DATA_QUBITS)), 
               chain(RESULT_QUBITS, DATA_QUBITS), inplace=True)
    
    # Step 3: Uncompute result qubits
    qc.compose(constraints_with_reuse().inverse(), inplace=True)
    
    return qc

if __name__ == "__main__":
    print(f"Grid size: {GRID_ROW_COUNT}×{GRID_COL_COUNT}")
    print(f"Data Qubits: {DATA_QUBIT_COUNT}")
    print(f"Helper Qubits (REUSED): {HELPER_QUBIT_COUNT}")
    print(f"Result Qubits (one per edge): {RESULT_QUBIT_COUNT}")
    print(f"Total qubits: {TOTAL_QUBIT_COUNT}")
    print(f"Savings: {16 - TOTAL_QUBIT_COUNT} qubits vs original!")
    
    grover_op = grover_operator(grover_oracle(), reflection_qubits=DATA_QUBITS)
    optimal_iterations_count = math.floor(math.pi / (4 * math.asin(math.sqrt(24 / 2**DATA_QUBIT_COUNT))))
    print(f"Grover iterations: {optimal_iterations_count}")
    
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT, DATA_QUBIT_COUNT)
    qc.h(DATA_QUBITS)
    qc.compose(grover_op.power(2), inplace=True)
    checkerboard_x(qc)
    qc.measure(DATA_QUBITS, range(DATA_QUBIT_COUNT))
    
    print("\nRunning simulation...")
    backend = Aer.get_backend("qasm_simulator")
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
    circuit_isa = pm.run(qc)
    result = backend.run(circuit_isa, shots=8192).result()
    counts = result.get_counts()
    
    valid_counts = {k: v for k, v in counts.items() if is_valid_grid(k)}
    total_valid = sum(valid_counts.values())
    total_shots = sum(counts.values())
    
    print(f"\nValid: {total_valid}/{total_shots} ({100*total_valid/total_shots:.1f}%)")
    print(f"Unique valid configs: {len(valid_counts)}")
    
    if total_valid / total_shots > 0.9:
        print("✓ SUCCESS: Still achieving >90% with ancilla reuse!")
    elif total_valid / total_shots > 0.7:
        print("⚠ PARTIAL: 70-90% success with ancilla reuse")
    else:
        print("✗ FAILURE: <70% success - ancilla reuse may have issues")
    
    if not valid_counts:
        print("WARNING: No valid solutions found!")
        valid_counts = counts
    
    best = max(valid_counts, key=valid_counts.get)
    tiles = [best[i:i+2] for i in range(0, len(best), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    
    print(f"\nBest solution: {decoded}")
    print(f"Is valid: {is_valid_grid(best)}")
    
    # Show top 5 results
    print(f"\nTop 5 results:")
    sorted_counts = sorted(valid_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for bitstring, count in sorted_counts:
        tiles_temp = [bitstring[i:i+2] for i in range(0, len(bitstring), 2)]
        decoded_temp = [TILE_MAP[t] for t in tiles_temp]
        print(f"  {decoded_temp}: {count} ({100*count/total_shots:.1f}%)")
    
    # Visualize
    grid = np.zeros((GRID_ROW_COUNT, GRID_COL_COUNT, 3))
    for i, tile in enumerate(decoded):
        grid[i // GRID_COL_COUNT, i % GRID_COL_COUNT] = COLOR_MAP[tile]
    
    fig, ax = plt.subplots(figsize=(max(6, GRID_COL_COUNT*2), max(6, GRID_ROW_COUNT*2)))
    ax.imshow(grid, interpolation='nearest')
    ax.axis("off")
    
    for i, tile in enumerate(decoded):
        r, c = i // GRID_COL_COUNT, i % GRID_COL_COUNT
        ax.text(c, r, f"{i}\n{tile}", ha='center', va='center', 
                fontsize=max(8, 12-max(GRID_ROW_COUNT,GRID_COL_COUNT)), 
                color='white', weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    for i in range(GRID_ROW_COUNT + 1):
        ax.axhline(i - 0.5, color='black', linewidth=2)
    for j in range(GRID_COL_COUNT + 1):
        ax.axvline(j - 0.5, color='black', linewidth=2)
    
    title_color = 'green' if is_valid_grid(best) else 'red'
    success_rate = 100*total_valid/total_shots
    ax.set_title(f"{GRID_ROW_COUNT}×{GRID_COL_COUNT} WFC (Ancilla Reuse)\n{TOTAL_QUBIT_COUNT} qubits, {success_rate:.1f}% valid", 
                 fontsize=14, color=title_color, weight='bold')
    
    plt.tight_layout()
    plt.show()