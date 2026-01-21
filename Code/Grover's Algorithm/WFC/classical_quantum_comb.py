import math
import random
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit_aer import Aer

# ======================================================
# CONFIG
# ======================================================
GRID_ROWS = 2
GRID_COLS = 3
MAX_MARKED = 5        # <<< limit oracle size here
SHOTS = 1

# ======================================================
# TILE DEFINITIONS
# ======================================================
TILE_MAP = {
    "00": "Water",
    "01": "Sand",
    "10": "Grass",
    "11": "Jungle",
}

COLOR_MAP = {
    "Water": (0.2, 0.4, 1.0),
    "Sand": (0.9, 0.85, 0.6),
    "Grass": (0.2, 0.7, 0.2),
    "Jungle": (0.0, 0.4, 0.0),
}

VALID = {
    "Water": {"Water", "Sand"},
    "Sand": {"Water", "Sand", "Grass"},
    "Grass": {"Sand", "Grass", "Jungle"},
    "Jungle": {"Grass", "Jungle"},
}

# ======================================================
# CLASSICAL GRID LOGIC
# ======================================================
def get_adjacencies(rows, cols):
    adj = []
    for r in range(rows):
        for c in range(cols - 1):
            adj.append((r * cols + c, r * cols + c + 1))
    for r in range(rows - 1):
        for c in range(cols):
            adj.append((r * cols + c, (r + 1) * cols + c))
    return adj


def is_valid_grid(bitstring, adjacencies):
    tiles = [bitstring[i:i+2] for i in range(0, len(bitstring), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    for a, b in adjacencies:
        if decoded[b] not in VALID[decoded[a]]:
            return False
    return True


def find_valid_bitstrings(rows, cols):
    adj = get_adjacencies(rows, cols)
    num_bits = rows * cols * 2

    valid = []
    for i in range(2 ** num_bits):
        bits = format(i, f"0{num_bits}b")
        if is_valid_grid(bits, adj):
            valid.append(bits)

    return valid, adj

# ======================================================
# GROVER ORACLE (FROM CLASSICAL SOLUTIONS)
# ======================================================
def grover_oracle_from_bitstrings(marked_states):
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)

    for target in marked_states:
        rev = target[::-1]
        zero_inds = [i for i, b in enumerate(rev) if b == "0"]

        if zero_inds:
            qc.x(zero_inds)

        qc.compose(
            MCMTGate(ZGate(), num_qubits - 1, 1),
            inplace=True
        )

        if zero_inds:
            qc.x(zero_inds)

    return qc

# ======================================================
# MAIN GROVER + AER RUN
# ======================================================
def run_wfc_grover(rows, cols):
    # ----------------------------------------------
    # CLASSICAL PREPROCESSING
    # ----------------------------------------------
    all_valid_states, adj = find_valid_bitstrings(rows, cols)

    if len(all_valid_states) > MAX_MARKED:
        marked_states = random.sample(all_valid_states, MAX_MARKED)
    else:
        marked_states = all_valid_states

    print(f"Grid: {rows}×{cols}")
    print(f"Total valid configurations (classical): {len(all_valid_states)}")
    print(f"Marked states used in oracle: {len(marked_states)}")

    # ----------------------------------------------
    # GROVER SETUP
    # ----------------------------------------------
    oracle = grover_oracle_from_bitstrings(marked_states)
    grover_op = grover_operator(oracle)

    num_qubits = oracle.num_qubits
    N = 2 ** num_qubits
    M = len(marked_states)

    # ----------------------------------------------
    # GEOMETRIC ITERATION SELECTION
    # ----------------------------------------------
    theta = math.asin(math.sqrt(M / N))
    max_k = 100

    iterations = max(
        range(max_k + 1),
        key=lambda k: math.sin((2 * k + 1) * theta) ** 2
    )

    best_prob = math.sin((2 * iterations + 1) * theta) ** 2

    print(
        f"Chosen iterations: {iterations} "
        f"(theoretical marked probability ≈ {100 * best_prob:.1f}%)"
    )

    # ----------------------------------------------
    # BUILD CIRCUIT
    # ----------------------------------------------
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    qc.compose(grover_op.power(iterations), inplace=True)
    qc.measure_all()

    # IMPORTANT for Aer
    qc = qc.decompose(reps=6)

    # ----------------------------------------------
    # RUN ON AER
    # ----------------------------------------------
    backend = Aer.get_backend("qasm_simulator")
    result = backend.run(qc, shots=SHOTS).result()
    counts = result.get_counts()

    # ----------------------------------------------
    # STATISTICS
    # ----------------------------------------------
    total_shots = sum(counts.values())
    marked_shots = sum(v for k, v in counts.items() if k in marked_states)
    marked_percent = 100 * marked_shots / total_shots

    print(f"Marked shots: {marked_shots}/{total_shots}")
    print(f"Marked percentage: {marked_percent:.2f}%")

    # Best measured marked state
    marked_counts = {k: v for k, v in counts.items() if k in marked_states}
    best = max(marked_counts, key=marked_counts.get)

    tiles = [best[i:i+2] for i in range(0, len(best), 2)]
    decoded = [TILE_MAP[t] for t in tiles]

    print("Best solution (marked):", decoded)
    print("Is classically valid:", is_valid_grid(best, adj))

    # ----------------------------------------------
    # GRID VISUALIZATION
    # ----------------------------------------------
    grid = np.zeros((rows, cols, 3))
    for i, tile in enumerate(decoded):
        grid[i // cols, i % cols] = COLOR_MAP[tile]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid)
    ax.axis("off")

    for i, tile in enumerate(decoded):
        r, c = i // cols, i % cols
        ax.text(
            c, r, tile,
            ha="center", va="center",
            color="white", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.6)
        )

    ax.set_title(
        f"{rows}×{cols} WFC (Grover + Aer)\n"
        f"Marked: {marked_percent:.1f}%",
        weight="bold"
    )
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------
    # HISTOGRAM
    # ----------------------------------------------
    plt.figure(figsize=(14, 5))
    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Counts")
    plt.xlabel("Bitstring")
    plt.title(
        f"Grover Measurement Histogram (Aer)\n"
        f"Marked shots: {marked_percent:.1f}%"
    )
    plt.tight_layout()
    plt.show()

    return qc, counts, decoded


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_wfc_grover(GRID_ROWS, GRID_COLS)
