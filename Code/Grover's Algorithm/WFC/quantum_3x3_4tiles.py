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
qc = QuantumCircuit(33, 18)

# --------------------------------------------------
# Data qubits (2 per cell)
# --------------------------------------------------
A0,A1 = 0,1
B0,B1 = 2,3
C0,C1 = 4,5
D0,D1 = 6,7
E0,E1 = 8,9
F0,F1 = 10,11
G0,G1 = 12,13
H0,H1 = 14,15
I0,I1 = 16,17

# Oracle flag
FLAG = 18

# Equality result ancillas (12)
AB,BC,DE,EF,GH,HI,AD,DG,BE,EH,CF,FI = range(19,31)

# Temporary XNOR ancillas
T0, T1 = 31, 32

# --------------------------------------------------
# 1️⃣ Superposition
# --------------------------------------------------
qc.h(range(18))

# --------------------------------------------------
# 2️⃣ Oracle: adjacency constraints
# --------------------------------------------------

pairs = [
    (A0,A1,B0,B1,AB), (B0,B1,C0,C1,BC),
    (D0,D1,E0,E1,DE), (E0,E1,F0,F1,EF),
    (G0,G1,H0,H1,GH), (H0,H1,I0,I1,HI),
    (A0,A1,D0,D1,AD), (D0,D1,G0,G1,DG),
    (B0,B1,E0,E1,BE), (E0,E1,H0,H1,EH),
    (C0,C1,F0,F1,CF), (F0,F1,I0,I1,FI),
]

for x0,x1,y0,y1,out in pairs:
    xnor(qc, x0, y0, T0)
    xnor(qc, x1, y1, T1)
    qc.ccx(T0, T1, out)
    xnor(qc, x1, y1, T1)
    xnor(qc, x0, y0, T0)

# AND all constraints → FLAG
qc.mcx(
    [AB,BC,DE,EF,GH,HI,AD,DG,BE,EH,CF,FI],
    FLAG
)

# Phase flip
qc.z(FLAG)

# --------------------------------------------------
# 3️⃣ Uncompute oracle
# --------------------------------------------------
qc.mcx(
    [AB,BC,DE,EF,GH,HI,AD,DG,BE,EH,CF,FI],
    FLAG
)

for x0,x1,y0,y1,out in reversed(pairs):
    xnor(qc, x0, y0, T0)
    xnor(qc, x1, y1, T1)
    qc.ccx(T0, T1, out)
    xnor(qc, x1, y1, T1)
    xnor(qc, x0, y0, T0)

# --------------------------------------------------
# 4️⃣ Diffusion operator (18 data qubits)
# --------------------------------------------------
qc.h(range(18))
qc.x(range(18))

qc.h(I1)
qc.mcx(list(range(17)), I1)
qc.h(I1)

qc.x(range(18))
qc.h(range(18))

# --------------------------------------------------
# 5️⃣ Measurement
# --------------------------------------------------
qc.measure(range(18), range(18))

# --------------------------------------------------
# Run
# --------------------------------------------------
sim = AerSimulator()
result = sim.run(qc, shots=512).result()
counts = result.get_counts()

print(counts)
plot_histogram(counts)
plt.show()
