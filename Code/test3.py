from qiskit import QuantumCircuit
from qiskit_quantuminspire.qi_provider import QIProvider
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


# --------------------------------------------------
# Helper: Toffoli decomposition (no ccx shortcut)
# --------------------------------------------------
def decomposed_toffoli(circ, q0, q1, q2):
    circ.h(q2)
    circ.cx(q1, q2)
    circ.tdg(q2)
    circ.cx(q0, q2)
    circ.t(q2)
    circ.cx(q1, q2)
    circ.tdg(q2)
    circ.cx(q0, q2)
    circ.t(q1)
    circ.t(q2)
    circ.cx(q0, q1)
    circ.h(q2)
    circ.t(q0)
    circ.tdg(q1)
    circ.cx(q0, q1)


# --------------------------------------------------
# Oracle for |110>
# --------------------------------------------------
def oracle(circ):
    circ.x(0)
    circ.h(2)
    decomposed_toffoli(circ, 0, 1, 2)
    circ.h(2)
    circ.x(0)


# --------------------------------------------------
# Diffusion operator
# --------------------------------------------------
def diffusion(circ):
    circ.h([0, 1, 2])
    circ.x([0, 1, 2])
    circ.h(2)
    decomposed_toffoli(circ, 0, 1, 2)
    circ.h(2)
    circ.x([0, 1, 2])
    circ.h([0, 1, 2])


# --------------------------------------------------
# Build Grover circuit
# --------------------------------------------------
circuit = QuantumCircuit(3, 3)

# Initialization
circuit.h([0, 1, 2])

# Two Grover iterations
oracle(circuit)
diffusion(circuit)

oracle(circuit)
diffusion(circuit)

# Measurement
circuit.measure([0, 1, 2], [0, 1, 2])


# --------------------------------------------------
# Run on Quantum Inspire
# --------------------------------------------------
provider = QIProvider()

# Show available backends
print(provider.backends())

backend = provider.get_backend(name="QX emulator")
job = backend.run(circuit, shots=1024)

result = job.result()
counts = result.get_counts()

print("Measurement results:")
print(counts)

plot_histogram(counts)
plt.show()
