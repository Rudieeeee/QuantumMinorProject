from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --------------------------------------------------
# Tile encoding:
# 00 = Water
# 01 = Sand
# 10 = Grass
# 11 = Jungle
# --------------------------------------------------

# --------------------------------------------------
# Helper: check equality of 2 qubits to fixed value
# --------------------------------------------------
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

# --------------------------------------------------
# Adjacency oracle for one edge
# --------------------------------------------------
def adjacency_oracle(qc, a0, a1, b0, b1, valid, t1, t2):

    # Water -> Water, Sand
    check_eq_2bit(qc, a0, a1, "00", t1)
    check_eq_2bit(qc, b0, b1, "00", t2)
    qc.ccx(t1, t2, valid)
    qc.reset(t2)
    check_eq_2bit(qc, b0, b1, "01", t2)
    qc.ccx(t1, t2, valid)
    qc.reset(t2)
    qc.reset(t1)

    # Sand -> Water, Sand, Grass
    check_eq_2bit(qc, a0, a1, "01", t1)
    for b in ["00", "01", "10"]:
        check_eq_2bit(qc, b0, b1, b, t2)
        qc.ccx(t1, t2, valid)
        qc.reset(t2)
    qc.reset(t1)

    # Grass -> Sand, Grass, Jungle
    check_eq_2bit(qc, a0, a1, "10", t1)
    for b in ["01", "10", "11"]:
        check_eq_2bit(qc, b0, b1, b, t2)
        qc.ccx(t1, t2, valid)
        qc.reset(t2)
    qc.reset(t1)

    # Jungle -> Grass, Jungle
    check_eq_2bit(qc, a0, a1, "11", t1)
    for b in ["10", "11"]:
        check_eq_2bit(qc, b0, b1, b, t2)
        qc.ccx(t1, t2, valid)
        qc.reset(t2)
    qc.reset(t1)

# --------------------------------------------------
# Grover Oracle (checks all adjacencies)
# --------------------------------------------------
def grover_oracle(qc, tiles, phase, valid, t1, t2):

    qc.x(phase)
    qc.h(phase)

    edges = [
        (0,1),(1,2),
        (3,4),(4,5),
        (6,7),(7,8),
        (0,3),(1,4),(2,5),
        (3,6),(4,7),(5,8)
    ]

    for i, j in edges:
        a0, a1 = tiles[i]
        b0, b1 = tiles[j]
        adjacency_oracle(qc, a0, a1, b0, b1, valid, t1, t2)
        qc.cz(valid, phase)
        qc.reset(valid)

    qc.h(phase)
    qc.x(phase)

# --------------------------------------------------
# Diffusion operator
# --------------------------------------------------
def diffusion(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)

# --------------------------------------------------
# Build circuit
# --------------------------------------------------
tile_qubits = QuantumRegister(18, "tile")
valid = QuantumRegister(1, "valid")
t1 = QuantumRegister(1, "t1")
t2 = QuantumRegister(1, "t2")
phase = QuantumRegister(1, "phase")
c = ClassicalRegister(18, "c")

qc = QuantumCircuit(tile_qubits, valid, t1, t2, phase, c)

tiles = [(tile_qubits[2*i], tile_qubits[2*i+1]) for i in range(9)]

for q in tile_qubits:
    qc.h(q)

grover_oracle(qc, tiles, phase[0], valid[0], t1[0], t2[0])
diffusion(qc, tile_qubits)

qc.measure(tile_qubits, c)

# --------------------------------------------------
# Run
# --------------------------------------------------
backend = Aer.get_backend("qasm_simulator")
job = backend.run(qc, shots=1024)
result = job.result()
counts = result.get_counts()

plot_histogram(counts)
plt.show()
