from PIL import Image
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

# ======================================================
# TILE DEFINITIONS
# ======================================================
TILES = {
    "00": "water",
    "01": "sand",
    "10": "grass",
    "11": "jungle",
}

COLORS = {
    "water": (30, 144, 255),
    "sand": (194, 178, 128),
    "grass": (34, 139, 34),
    "jungle": (0, 100, 0),
}

CELL_SIZE = 100

# ======================================================
# FORBIDDEN ADJACENCIES (DIRECTED)
# ======================================================
FORBIDDEN = {
    "00": ["10", "11"],  # water ! grass, jungle
    "01": ["11"],        # sand ! jungle
    "10": ["00"],        # grass ! water
    "11": ["00", "01"],  # jungle ! water, sand
}

# ======================================================
# QUANTUM HELPERS
# ======================================================
def check_eq_2bit(qc, q0, q1, val, anc):
    if val[0] == "0":
        qc.x(q0)
    if val[1] == "0":
        qc.x(q1)
    qc.ccx(q0, q1, anc)
    if val[0] == "0":
        qc.x(q0)
    if val[1] == "0":
        qc.x(q1)

def check_forbidden_pair(qc, a0, a1, b0, b1, bad, t1, t2):
    for a_val, b_list in FORBIDDEN.items():
        check_eq_2bit(qc, a0, a1, a_val, t1)
        for b_val in b_list:
            check_eq_2bit(qc, b0, b1, b_val, t2)
            qc.ccx(t1, t2, bad)  # bad = bad OR (t1 AND t2)
            qc.reset(t2)
        qc.reset(t1)

# ======================================================
# GROVER ORACLE (CORRECT)
# ======================================================
def grover_oracle(qc, tiles, phase, bad, t1, t2):

    qc.x(phase)
    qc.h(phase)

    edges = [(0,1), (2,3), (0,2), (1,3)]

    # Accumulate forbidden edges (NO reset of bad here)
    for i, j in edges:
        a0, a1 = tiles[i]
        b0, b1 = tiles[j]

        # check both directions
        check_forbidden_pair(qc, a0, a1, b0, b1, bad, t1, t2)
        check_forbidden_pair(qc, b0, b1, a0, a1, bad, t1, t2)

    # Phase flip only if bad == 0
    qc.x(bad)
    qc.cz(bad, phase)
    qc.x(bad)

    qc.reset(bad)

    qc.h(phase)
    qc.x(phase)

# ======================================================
# DIFFUSION
# ======================================================
def diffusion(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)

# ======================================================
# BUILD & RUN
# ======================================================
tile_qubits = QuantumRegister(8, "tile")
bad = QuantumRegister(1, "bad")
t1 = QuantumRegister(1, "t1")
t2 = QuantumRegister(1, "t2")
phase = QuantumRegister(1, "phase")
c = ClassicalRegister(8, "c")

qc = QuantumCircuit(tile_qubits, bad, t1, t2, phase, c)

tiles = [(tile_qubits[2*i], tile_qubits[2*i+1]) for i in range(4)]

for q in tile_qubits:
    qc.h(q)

grover_oracle(qc, tiles, phase[0], bad[0], t1[0], t2[0])
diffusion(qc, tile_qubits)

qc.measure(tile_qubits, c)

backend = Aer.get_backend("qasm_simulator")
counts = backend.run(qc, shots=1024).result().get_counts()

# ======================================================
# PICK MOST PROBABLE
# ======================================================
best_state = max(counts, key=counts.get)
bits = best_state[::-1]

decoded = [TILES[bits[i:i+2]] for i in range(0, 8, 2)]

# ======================================================
# RENDER
# ======================================================
img = Image.new("RGB", (2*CELL_SIZE, 2*CELL_SIZE))
idx = 0
for y in range(2):
    for x in range(2):
        color = COLORS[decoded[idx]]
        for dy in range(CELL_SIZE):
            for dx in range(CELL_SIZE):
                img.putpixel((x*CELL_SIZE+dx, y*CELL_SIZE+dy), color)
        idx += 1

img.show()
img.save("quantum_wfc_2x2_STRICT.png")

plt.bar(counts.keys(), counts.values())
plt.xticks(rotation=90)
plt.title("Strict forbidden-pair oracle")
plt.show()
