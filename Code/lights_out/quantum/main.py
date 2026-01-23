import math
import os
import numpy as np
from pathlib import Path
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.primitives import StatevectorSampler
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

# from qiskit_quantuminspire import cqasm as qi_cqasm
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeTorino, FakeMarrakesh
from qiskit.transpiler import generate_preset_pass_manager
from dotenv import load_dotenv

# load_dotenv()
# QiskitRuntimeService.save_account(
#     token=os.environ["API_KEY"],
#     instance=os.environ["CRN"],
# )


# def export_cqasm(
#     circuit: QuantumCircuit,
#     output_path: str = "ciruit.cq",
#     backend_name: str | None = None,
# ) -> None:
#     """Export a Qiskit circuit to cQASM (Quantum Inspire) and write it to disk."""
#     if not backend_name:
#         raise ValueError(
#             "backend_name is required (e.g. backend_name='Tuna-9'). "
#             "This exporter transpiles against a specific Quantum Inspire backend."
#         )

#     # qiskit-quantuminspire's cQASM exporter relies on qubit indices; if you use
#     # multiple QuantumRegisters, Qiskit's per-register indices can collide (e.g.
#     # v[0] and t[0] both have index 0). Flatten to a single register first.
#     flattened = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
#     flattened.compose(
#         circuit, qubits=flattened.qubits, clbits=flattened.clbits, inplace=True
#     )

#     from qiskit_quantuminspire.qi_provider import QIProvider

#     backend = QIProvider().get_backend(backend_name)

#     transpiled = transpile(flattened, backend=backend, optimization_level=3)
#     cqasm_str = qi_cqasm.dumps(transpiled)

#     Path(output_path).write_text(cqasm_str, encoding="utf-8")


def generate_conditions(size):
    """
    Generate conditions for Lights Out game.

    Args:
        size: The size of the grid (e.g., 2 for a 2x2 grid, 3 for 3x3, etc.)

    Returns:
        List of tuples: (list_of_indices, temp_qubit_index)
        Each tuple represents which variable qubits affect which temp qubit
    """
    conditions = []

    for y in range(size):
        for x in range(size):
            i = y * size + x
            adjacent = [i]  # Include the light itself

            # Check all 4 adjacent positions (right, down, left, up)
            offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]

            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    adjacent.append(ny * size + nx)

            # Each condition is (list of var qubits, temp qubit index)
            conditions.append((adjacent, i))

    return conditions


def add_oracle(
    qc, var_qubits, temp_qubits, output_qubit, conditions, ancilla_qubits=None
):
    # Optimize layout by grouping CNOTs by their relative offset.
    # This ensures that gates operating on disjoint sets of qubits are added
    # in "layers", helping the compiler schedule them in parallel to reduce circuit depth.
    layers = {}
    for vars, temp in conditions:
        for var in vars:
            diff = var - temp
            if diff not in layers:
                layers[diff] = []
            layers[diff].append((var, temp))

    # Sort offsets to apply layers in a consistent order (e.g. self, then neighbors)
    # This structure (v-t) naturally groups: Center, Right, Left, Down, Up interactions
    sorted_offsets = sorted(layers.keys())

    # Forward CNOTs (Compute)
    for offset in sorted_offsets:
        for var, temp in layers[offset]:
            qc.cx(var_qubits[var], temp_qubits[temp])

    # Use MCX. The transpiler's HighLevelSynthesis pass will automatically
    # use the available ancilla qubits in the circuit for an optimal decomposition
    # (e.g. v-chain) when optimization_level=3 is used.
    qc.mcx(temp_qubits, output_qubit, ancilla_qubits=ancilla_qubits, mode="v-chain")

    # Reverse CNOTs (Uncompute) - mirror order for symmetry
    for offset in sorted_offsets[::-1]:
        for var, temp in layers[offset]:
            qc.cx(var_qubits[var], temp_qubits[temp])


def add_grover_diffusion(qc, var_qubits):
    qc.h(var_qubits)
    qc.x(var_qubits)
    qc.h(var_qubits[-1])
    qc.mcx(var_qubits[:-1], var_qubits[-1])
    qc.h(var_qubits[-1])
    qc.x(var_qubits)
    qc.h(var_qubits)


def create_lights_out_circuit(size, initial_state, num_iterations=3):
    """
    Create a quantum circuit for solving Lights Out game using Grover's algorithm.

    Args:
        size: The size of the grid (e.g., 2 for a 2x2 grid, 3 for 3x3, etc.)
        initial_state: List of 0s and 1s representing initial light configuration
                      (0 = light off, 1 = light on)
        num_iterations: Number of Grover iterations (default: 3)

    Returns:
        Tuple of (quantum_circuit, var_qubits, cbits)
    """
    n = size * size

    # Create quantum registers
    var_qubits = QuantumRegister(n, name="v")
    temp_qubits = QuantumRegister(n, name="t")
    # For v-chain decomposition of MCX with n controls, we need n-2 ancillas (for n>=3)
    # Using a few ancillas generally helps depth on hardware topologies.
    num_ancillas = max(0, n - 2)
    ancilla_qubits = (
        QuantumRegister(num_ancillas, name="anc") if num_ancillas > 0 else None
    )

    output_qubit = QuantumRegister(1, name="out")
    cbits = ClassicalRegister(n, name="cbits")

    # Generate conditions for the grid
    conditions = generate_conditions(size)

    # Quantum circuit
    registers = [var_qubits, temp_qubits]
    if ancilla_qubits:
        registers.append(ancilla_qubits)
    registers.append(output_qubit)
    registers.append(cbits)

    qc = QuantumCircuit(*registers)

    # Set initial state based on which lights are off (0)
    for i, state in enumerate(initial_state[::-1]):
        if state == 0:
            qc.x(temp_qubits[i])

    # Create superposition of all possible button press combinations
    qc.h(var_qubits)

    # Prepare output qubit in |−⟩ state for phase kickback
    qc.x(output_qubit)
    qc.h(output_qubit)

    for _ in range(num_iterations):
        add_oracle(
            qc, var_qubits, temp_qubits, output_qubit, conditions, ancilla_qubits
        )
        add_grover_diffusion(qc, var_qubits)

    # Measure the variable qubits
    qc.measure(var_qubits, cbits)

    return qc, var_qubits, cbits


def create_circuit_from_matrix(initial_state_matrix):
    """
    Create a quantum circuit for solving Lights Out from a 2D matrix.

    Args:
        initial_state_matrix: 2D numpy array or list of lists with 0s and 1s
                             (0 = light off, 1 = light on)

    Returns:
        Tuple of (quantum_circuit, var_qubits, cbits, size)
    """
    # Convert to numpy array if needed
    state_array = np.array(initial_state_matrix)

    # Determine size and flatten
    size = state_array.shape[0]
    initial_state = state_array.flatten()

    # Calculate optimal number of Grover iterations
    # num_iterations = math.floor(
    #     math.pi / (4 * math.asin(1 / math.sqrt(2 ** (size**2))))
    # )
    num_iterations = 3

    # Create the circuit
    qc, var_qubits, cbits = create_lights_out_circuit(
        size, initial_state, num_iterations
    )

    # Print circuit statistics
    total_qubits = qc.num_qubits
    num_gates = qc.size()

    print(f"Grid size: {size}x{size}")
    print(f"Total qubits required: {total_qubits}")
    print(f"Total gates in circuit: {num_gates}")
    print(f"Grover iterations: {num_iterations}")

    return qc, var_qubits, cbits, size


def run_circuit(qc, shots=10000):
    """
    Run a quantum circuit and return the measurement counts.

    Args:
        qc: The quantum circuit to run
        shots: Number of measurement shots (default: 10000)

    Returns:
        Dictionary of measurement counts
    """
    sampler = StatevectorSampler()
    result = sampler.run([qc], shots=shots).result()
    counts = result[0].data.cbits.get_counts()

    return counts


def run_on_ibm_simulator(qc, shots=10000):
    """
    Run a quantum circuit on an IBM fake simulator

    Args:
        qc: QuantumCircuit to run
        shots: Number of measurement shots

    Returns:
        Measurement counts (dict)
    """
    backend = FakeMarrakesh()

    transpiled = transpile(qc, backend=backend, optimization_level=3)
    print("Transpiled")
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    return counts


def plot_top_results(counts, top_n=20, filename="plot"):
    """
    Plot the top measurement results from a quantum circuit run.

    Args:
        counts: Dictionary of measurement counts
        top_n: Number of top results to display (default: 20)
        filename: Output filename for the plot (default: "plot")

    Returns:
        None (saves plot to file)
    """
    # Ensure top_n doesn't exceed available results
    top_n = min(top_n, len(counts))

    # Filter to show only top results
    sorted_counts = dict(
        sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    print(f"Top {top_n} results out of {len(counts)} total outcomes:")
    for state, count in sorted_counts.items():
        print(f"  {state}: {count}")

    # Create a readable plot with only top results
    plot_histogram(sorted_counts, figsize=(12, 6), bar_labels=True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")


def run_on_ibm_hardware(qc, shots=5_000, min_qubits=None):
    service = QiskitRuntimeService()

    backend = service.least_busy(
        simulator=False, operational=True, min_num_qubits=min_qubits
    )
    # backend = service.backend("ibm_marrakesh")
    print("Will run on " + backend.name)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_circuit = pm.run(qc)

    from qiskit.transpiler import PassManager, InstructionDurations
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
    from qiskit.circuit.library import XGate

    # Retrieve backend constraints
    durations = InstructionDurations.from_backend(backend)
    # Try to get pulse alignment
    pulse_alignment = backend.target.timing_constraints().pulse_alignment

    # Use a simple XY4 or XX sequence. X-X is identity and robust against T2.
    dd_sequence = [XGate(), XGate()]

    # We must analyze scheduling (ALAP) before padding with DD
    dd_pm = PassManager(
        [
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(
                durations, dd_sequence, pulse_alignment=pulse_alignment
            ),
        ]
    )

    # Apply DD to the transpiled (ISA) circuit
    isa_circuit = dd_pm.run(isa_circuit)
    print("Applied Dynamical Decoupling (X-X) to idle qubits.")

    isa_circuit.draw(output="mpl", idle_wires=False, style="iqp").savefig("ibm-circuit")
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    print("Running the sampler...")
    result = sampler.run([isa_circuit]).result()
    counts = result[0].data.cbits.get_counts()

    return counts


# 0 - turned off, 1 - turned on
initial_state_matrix = [
    [1, 1, 0],
    [1, 0, 1],
    [1, 0, 1],
]
# Solution: 101 011 100

# initial_state_matrix = [
#     [1, 0],
#     [0, 1],
# ]

# Create circuit from matrix
qc, var_qubits, cbits, size = create_circuit_from_matrix(initial_state_matrix)

# Draw and save the circuit
qc.draw("mpl").savefig("circuit")

# Export to cQASM (Quantum Inspire)
# try:
#     export_cqasm(qc, "ciruit.cq", backend_name="Tuna-9")
#     print("Wrote cQASM to ciruit.cq")
# except Exception as exc:
#     print(f"cQASM export failed: {exc}")


# Run the circuit
# counts = run_on_ibm_simulator(qc, shots=1000)
# counts = run_on_ibm_hardware(qc)
counts = run_circuit(qc, shots=1_000)

# Plot the top results
plot_top_results(counts, top_n=20, filename="plot")
