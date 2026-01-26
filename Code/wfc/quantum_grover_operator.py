from itertools import chain
from random import getrandbits
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit_aer import Aer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import matplotlib.pyplot as plt

GRID_ROW_COUNT = 3
GRID_COL_COUNT = 3
GRID_TILE_COUNT = GRID_ROW_COUNT * GRID_COL_COUNT
GRID_EDGES = []
for row in range(GRID_ROW_COUNT):
    for col in range(GRID_COL_COUNT - 1):
        a = GRID_COL_COUNT * row + col
        b = GRID_COL_COUNT * row + col + 1
        GRID_EDGES.append((a, b))
        GRID_EDGES.append((b, a))
for row in range(GRID_ROW_COUNT - 1):
    for col in range(GRID_COL_COUNT):
        a = GRID_COL_COUNT * row + col
        b = GRID_COL_COUNT * (row + 1) + col
        GRID_EDGES.append((a, b))
        GRID_EDGES.append((b, a))
        
DATA_QUBIT_COUNT = GRID_TILE_COUNT * 2
MIN_ANCLILLA_QUBIT_COUNT = math.ceil((math.sqrt(8 * len(GRID_EDGES) - 7) + 1) / 2)
ANCILLA_QUBIT_COUNT = MIN_ANCLILLA_QUBIT_COUNT
TOTAL_QUBIT_COUNT = DATA_QUBIT_COUNT + ANCILLA_QUBIT_COUNT
    
DATA_QUBITS = list(range(DATA_QUBIT_COUNT))
ANCILLA_QUBITS = list(range(DATA_QUBIT_COUNT, DATA_QUBIT_COUNT + ANCILLA_QUBIT_COUNT))

GROVER_ITERATIONS = 9
SHOT_COUNT = 1024

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

def constraints():
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT)
    reserved_ancilla_count = 0
    i = 0
    while True:
        qc_part = QuantumCircuit(TOTAL_QUBIT_COUNT)
        used_ancillas = []

        for j in range(ANCILLA_QUBIT_COUNT - reserved_ancilla_count - 1):
            if i >= len(GRID_EDGES):
                break

            a, b = GRID_EDGES[i]
            data = [DATA_QUBITS[a*2], DATA_QUBITS[a*2+1], DATA_QUBITS[b*2+1]]
            ancilla = ANCILLA_QUBITS[j]

            qc_part.mcx(data, ancilla)
            qc_part.x(data)
            qc_part.mcx(data, ancilla)
            qc_part.x(data)
            qc_part.x(ancilla)

            i += 1
            used_ancillas.append(ancilla)
        
        qc.compose(qc_part, inplace=True)

        if i >= len(GRID_EDGES):
            break

        reserved_ancilla_count += 1
        qc.mcx(used_ancillas, ANCILLA_QUBITS[-reserved_ancilla_count])
        qc.compose(qc_part.inverse(), inplace=True)
    
    used_ancillas.extend(ANCILLA_QUBITS[-reserved_ancilla_count:])

    return qc, used_ancillas

def grover_oracle():
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT)
    constraints_qc, used_ancillas = constraints()
    qc.compose(constraints_qc, inplace=True)
    qc.compose(MCMTGate(ZGate(), len(used_ancillas), len(DATA_QUBITS)), chain(used_ancillas, DATA_QUBITS), inplace=True)
    qc.compose(constraints_qc.inverse(), inplace=True)
    return qc

def estimate_optimal_iterations(valid_probability):
    p0 = 0
    for i in range(SHOT_COUNT):
        if is_valid_grid(f"{getrandbits(DATA_QUBIT_COUNT):0{DATA_QUBIT_COUNT}b}"):
            p0 += 1
    p0 /= SHOT_COUNT

    a = math.acos(1 - 2 * p0)
    b = math.acos(1 - 2 * valid_probability)
    c = GROVER_ITERATIONS * math.pi * (1 - a / math.pi)

    esitmate_up = c / (b - a)
    estimate_down = c / (math.tau - b - a)

    return esitmate_up, estimate_down

if __name__ == "__main__":
    print(f"Grid size: {GRID_ROW_COUNT}×{GRID_COL_COUNT}")
    print(f"Data Qubits: {DATA_QUBIT_COUNT}")
    print(f"Ancilla Qubits: {ANCILLA_QUBIT_COUNT}")
    print(f"Total qubits: {TOTAL_QUBIT_COUNT}")
    
    grover_op = grover_operator(grover_oracle(), reflection_qubits=DATA_QUBITS)
    print(f"Grover iterations: {GROVER_ITERATIONS}")
    
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT, DATA_QUBIT_COUNT)
    qc.h(DATA_QUBITS)
    qc.compose(grover_op.power(GROVER_ITERATIONS), inplace=True)
    checkerboard_x(qc)
    qc.measure(DATA_QUBITS, range(DATA_QUBIT_COUNT))

    print(f"\nCircuit depth before transpilation: {qc.depth()}")
    print("Transpiling circuit...")
    backend = Aer.get_backend("qasm_simulator")
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
    circuit_isa = pm.run(qc)
    print(f"Circuit depth after transpilation: {circuit_isa.depth()}")
    
    print("\nRunning simulation...")
    result = backend.run(circuit_isa, shots=SHOT_COUNT).result()
    counts = result.get_counts()
    
    valid_counts = {k: v for k, v in counts.items() if is_valid_grid(k)}
    total_valid = sum(valid_counts.values())
    
    valid_probability = total_valid / SHOT_COUNT
    print(f"\nValid: {total_valid}/{SHOT_COUNT} ({valid_probability*100:.1f}%)")
    print(f"Unique valid configs: {len(valid_counts)}")
    
    if not valid_counts:
        print("WARNING: No valid solutions found!")
        valid_counts = counts
    
    best = max(valid_counts, key=valid_counts.get)
    tiles = [best[i:i+2] for i in range(0, len(best), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    
    print(f"Best solution: {decoded}")
    print(f"Is valid: {is_valid_grid(best)}")

    esitmate_up, estimate_down = estimate_optimal_iterations(valid_probability)
    print("\nOptimal Grover iteration estimates:")
    print(f"If optimum is above {GROVER_ITERATIONS}: {esitmate_up:.2f}")
    print(f"If optimum is below {GROVER_ITERATIONS}: {estimate_down:.2f}")
    
    grid = np.zeros((GRID_ROW_COUNT, GRID_COL_COUNT, 3))
    for i, tile in enumerate(decoded):
        grid[i // GRID_COL_COUNT, i % GRID_COL_COUNT] = COLOR_MAP[tile]
    
    fig, ax = plt.subplots(figsize=(max(6, GRID_COL_COUNT*2), max(6, GRID_ROW_COUNT*2)))
    ax.imshow(grid, interpolation='nearest')
    ax.axis("off")
    
    for i, tile in enumerate(decoded):
        r, c = i // GRID_COL_COUNT, i % GRID_COL_COUNT
        ax.text(c, r, f"{i}\n{tile}", ha='center', va='center', fontsize=max(8, 12-max(GRID_ROW_COUNT,GRID_COL_COUNT)), 
                color='white', weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    for i in range(GRID_ROW_COUNT + 1):
        ax.axhline(i - 0.5, color='black', linewidth=2)
    for j in range(GRID_COL_COUNT + 1):
        ax.axvline(j - 0.5, color='black', linewidth=2)
    
    title_color = 'green' if is_valid_grid(best) else 'red'
    if GRID_ROW_COUNT <= 3 and GRID_COL_COUNT <= 3:
        grid_str = '\n'.join([str(decoded[row*GRID_COL_COUNT:(row+1)*GRID_COL_COUNT]) for row in range(GRID_ROW_COUNT)])
        ax.set_title(f"{GRID_ROW_COUNT}×{GRID_COL_COUNT} WFC Grid\n{grid_str}", fontsize=11, color=title_color, weight='bold')
    else:
        ax.set_title(f"{GRID_ROW_COUNT}×{GRID_COL_COUNT} WFC Grid", fontsize=14, color=title_color, weight='bold')
    
    plt.tight_layout()
    plt.show()