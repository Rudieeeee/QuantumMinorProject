from qiskit import QuantumCircuit

from qiskit_quantuminspire.qi_provider import QIProvider

provider = QIProvider()

# Show all current supported backends:
print(provider.backends())

circuit = QuantumCircuit(3, 10)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)

backend = provider.get_backend(name="QX emulator")
job = backend.run(circuit, shots=1024)

result = job.result()
print(result)