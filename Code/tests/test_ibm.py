from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt

# ONE-TIME SETUP: Uncomment and run once to save your token
# QiskitRuntimeService.save_account(
#     channel="ibm_quantum_platform",
#     token="YOUR_TOKEN_HERE",
#     overwrite=True
# )

print("=" * 60)
print("QUANTUM 'HELLO WORLD' - BELL STATE")
print("=" * 60)

# Connect to IBM Quantum
print("\n[1/5] Connecting to IBM Quantum...")
try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    print("✓ Connected successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nMake sure you've saved your token first!")
    exit(1)

# List available backends
print("\n[2/5] Available quantum backends:")
backends = service.backends(operational=True, simulator=False)
for backend in backends:
    status = backend.status()
    print(f"  • {backend.name}: {backend.num_qubits} qubits, Queue: {status.pending_jobs} jobs")

# Select a backend (just need 2 qubits for Bell state)
print("\n[3/5] Selecting backend...")
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
print(f"✓ Selected: {backend.name} ({backend.num_qubits} qubits)")
print(f"  Queue: {backend.status().pending_jobs} jobs waiting")

# Create a Bell State circuit (entanglement = quantum "Hello World")
print("\n[4/5] Creating Bell State circuit...")
qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits

# Apply Hadamard gate to qubit 0 (creates superposition)
qc.h(0)

# Apply CNOT gate (creates entanglement)
qc.cx(0, 1)

# Measure both qubits
qc.measure([0, 1], [0, 1])

print("✓ Circuit created!")
print("\nCircuit diagram:")
print(qc.draw(output='text'))

# Get classical register name
classical_register_name = qc.cregs[0].name

# Transpile for hardware
print("\n[5/5] Transpiling and running on quantum hardware...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
circuit_isa = pm.run(qc)
print(f"✓ Transpiled circuit depth: {circuit_isa.depth()} gates")

# Run on quantum hardware
shots = 1024
print(f"\nSubmitting job with {shots} shots...")
sampler = Sampler(backend)
job = sampler.run([circuit_isa], shots=shots)

print(f"✓ Job submitted!")
print(f"  Job ID: {job.job_id()}")
print(f"  Status: https://quantum.ibm.com/jobs/{job.job_id()}")
print("\n  Waiting for results...")

# Get results
try:
    result = job.result()
    pub_result = result[0]
    counts = getattr(pub_result.data, classical_register_name).get_counts()
    print("✓ Results received!")
except Exception as e:
    print(f"✗ Job failed: {e}")
    exit(1)

# Display results
print("\n" + "=" * 60)
print("RESULTS - BELL STATE MEASUREMENTS")
print("=" * 60)

# Bell state should give ~50% |00⟩ and ~50% |11⟩ (entangled!)
total = sum(counts.values())
print(f"\nTotal measurements: {total}")
print("\nMeasurement outcomes:")

for bitstring in ['00', '01', '10', '11']:
    count = counts.get(bitstring, 0)
    percentage = (count / total) * 100
    bar = '█' * int(percentage / 2)
    print(f"  |{bitstring}⟩: {count:4d} ({percentage:5.1f}%) {bar}")

# Check if it's a good Bell state
count_00 = counts.get('00', 0)
count_11 = counts.get('11', 0)
count_01 = counts.get('01', 0)
count_10 = counts.get('10', 0)
entangled_percentage = ((count_00 + count_11) / total) * 100

print(f"\n{'=' * 60}")
print("ANALYSIS")
print(f"{'=' * 60}")
print(f"Entangled states (|00⟩ + |11⟩): {entangled_percentage:.1f}%")
print(f"Error states (|01⟩ + |10⟩): {100 - entangled_percentage:.1f}%")

if entangled_percentage > 90:
    print("\n✓ EXCELLENT! Strong quantum entanglement observed!")
elif entangled_percentage > 70:
    print("\n✓ GOOD! Quantum entanglement detected (some hardware noise)")
elif entangled_percentage > 50:
    print("\n⚠ MODERATE: Entanglement present but significant noise")
else:
    print("\n✗ HIGH NOISE: Results heavily affected by quantum errors")

print("\nWhat this means:")
print("• In a Bell state, measuring qubit 0 instantly determines qubit 1")
print("• Perfect entanglement → only |00⟩ and |11⟩ outcomes")
print(f"• Your quantum computer achieved {entangled_percentage:.1f}% fidelity")
print(f"• The {100 - entangled_percentage:.1f}% errors are from real quantum hardware noise!")

# Visualize
print("\n[6/5] Generating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart of results
states = ['00', '01', '10', '11']
values = [counts.get(s, 0) for s in states]
colors = ['green' if s in ['00', '11'] else 'red' for s in states]

ax1.bar(states, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_xlabel('Quantum State', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title(f'Bell State Results - {backend.name}\n{shots} shots', 
              fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (state, value) in enumerate(zip(states, values)):
    percentage = (value / total) * 100
    ax1.text(i, value, f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Circuit diagram
ax2.axis('off')
circuit_text = str(qc.draw(output='text'))
ax2.text(0.1, 0.5, circuit_text, fontfamily='monospace', fontsize=10, 
         verticalalignment='center', transform=ax2.transAxes)
ax2.set_title('Bell State Circuit', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("✓ QUANTUM HELLO WORLD COMPLETE!")
print("=" * 60)
print(f"\nYou just ran a real quantum circuit on {backend.name}!")
print("This demonstrated quantum superposition and entanglement.")
print(f"Job ID: {job.job_id()}")