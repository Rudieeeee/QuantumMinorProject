from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --------------------------------------------------
# XNOR helper
# --------------------------------------------------
def xnor(qc, a, b, out):
    qc.cx(a, out)
    qc.cx(b, out)
    qc.x(out)

# --------------------------------------------------
# Circuit
# --------------------------------------------------
qc = QuantumCircuit(16, 8)

# Data qubits (2 per cell)
A0, A1 = 0, 1
B0, B1 = 2, 3
C0, C1 = 4, 5
D0, D1 = 6, 7

# Oracle flag
FLAG = 8

# Per-constraint equality results
AB, AC, BD, CD = 9, 10, 11, 12

# Temporary XNOR ancillas
T0, T1 = 13, 14

# AND helper
ANDTMP = 15

# --------------------------------------------------
# 1️⃣ Superposition
# --------------------------------------------------
qc.h(range(8))

# --------------------------------------------------
# 2️⃣ Oracle: adjacency constraints
# --------------------------------------------------

# ---- A == B ----
xnor(qc, A0, B0, T0)
xnor(qc, A1, B1, T1)
qc.ccx(T0, T1, AB)
xnor(qc, A1, B1, T1)
xnor(qc, A0, B0, T0)

# ---- A == C ----
xnor(qc, A0, C0, T0)
xnor(qc, A1, C1, T1)
qc.ccx(T0, T1, AC)
xnor(qc, A1, C1, T1)
xnor(qc, A0, C0, T0)

# ---- B == D ----
xnor(qc, B0, D0, T0)
xnor(qc, B1, D1, T1)
qc.ccx(T0, T1, BD)
xnor(qc, B1, D1, T1)
xnor(qc, B0, D0, T0)

# ---- C == D ----
xnor(qc, C0, D0, T0)
xnor(qc, C1, D1, T1)
qc.ccx(T0, T1, CD)
xnor(qc, C1, D1, T1)
xnor(qc, C0, D0, T0)

# AND all constraints → FLAG
qc.mcx([AB, AC, BD, CD], FLAG)

# Phase flip
qc.z(FLAG)

# --------------------------------------------------
# 3️⃣ Uncompute oracle
# --------------------------------------------------
qc.mcx([AB, AC, BD, CD], FLAG)

# Uncompute constraints (reverse order)
for (X0, X1, Y0, Y1, OUT) in [
    (C0, C1, D0, D1, CD),
    (B0, B1, D0, D1, BD),
    (A0, A1, C0, C1, AC),
    (A0, A1, B0, B1, AB),
]:
    xnor(qc, X0, Y0, T0)
    xnor(qc, X1, Y1, T1)
    qc.ccx(T0, T1, OUT)
    xnor(qc, X1, Y1, T1)
    xnor(qc, X0, Y0, T0)

# --------------------------------------------------
# 4️⃣ Diffusion
# --------------------------------------------------
qc.h(range(8))
qc.x(range(8))

qc.h(7)
qc.mcx(list(range(7)), 7)
qc.h(7)

qc.x(range(8))
qc.h(range(8))

# --------------------------------------------------
# 5️⃣ Measurement
# --------------------------------------------------
qc.measure(range(8), range(8))

# --------------------------------------------------
# Run
# --------------------------------------------------
sim = AerSimulator()
result = sim.run(qc, shots=2000).result()
counts = result.get_counts()

print(counts)
plot_histogram(counts)
plt.show()
