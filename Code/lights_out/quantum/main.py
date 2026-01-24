import math
import os
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.primitives import StatevectorSampler
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
from qiskit.transpiler import (
    generate_preset_pass_manager,
    PassManager,
)
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit.circuit.library import XGate
from dotenv import load_dotenv


def init_ibm_credentials():
    # This needs to be run only once
    load_dotenv()
    QiskitRuntimeService.save_account(
        token=os.environ["API_KEY"],
        instance=os.environ["CRN"],
    )


def generate_conditions(size):
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

            conditions.append((adjacent, i))

    return conditions


# https://github.com/Qiskit/qiskit/blob/stable/2.3/qiskit/synthesis/multi_controlled/mcx_synthesis.py#L69-L109
# Improves mcx peformance on hardware
def mcx(qc, control_qubits, target_qubits, anncilla_qubits):
    qc.rccx(control_qubits[0], control_qubits[1], anncilla_qubits[0])
    i = 0
    for j in range(2, control_qubits.size - 1):
        qc.rccx(control_qubits[j], anncilla_qubits[i], anncilla_qubits[i + 1])
        i += 1

    qc.ccx(control_qubits[-1], anncilla_qubits[i], target_qubits)

    for j in reversed(range(2, control_qubits.size - 1)):
        qc.rccx(control_qubits[j], anncilla_qubits[i - 1], anncilla_qubits[i])
        i -= 1

    qc.rccx(control_qubits[0], control_qubits[1], anncilla_qubits[i])


def add_oracle(qc, var_qubits, temp_qubits, output_qubit, conditions, ancilla_qubits):
    layers = {}
    for vars, temp in conditions:
        for var in vars:
            diff = var - temp
            if diff not in layers:
                layers[diff] = []
            layers[diff].append((var, temp))

    # Possibly improves performance on hardware
    sorted_offsets = sorted(layers.keys())

    # Compute
    for offset in sorted_offsets:
        for var, temp in layers[offset]:
            qc.cx(var_qubits[var], temp_qubits[temp])

    mcx(qc, temp_qubits, output_qubit, ancilla_qubits)

    # Uncompute
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
    n = size * size

    var_qubits = QuantumRegister(n, name="v")
    temp_qubits = QuantumRegister(n, name="t")

    # Required for mcx
    ancilla_qubits = QuantumRegister(n - 2, name="anc")

    output_qubit = QuantumRegister(1, name="out")
    cbits = ClassicalRegister(n, name="cbits")

    conditions = generate_conditions(size)

    qc = QuantumCircuit(var_qubits, temp_qubits, ancilla_qubits, output_qubit, cbits)

    # Set initial state based on which lights are off (i.e. 0)
    # Additionally, reverse the bits
    for i, state in enumerate(initial_state[::-1]):
        if state == 0:
            qc.x(temp_qubits[i])

    qc.h(var_qubits)

    qc.x(output_qubit)
    qc.h(output_qubit)

    # Grover's algorithmn
    for _ in range(num_iterations):
        add_oracle(
            qc, var_qubits, temp_qubits, output_qubit, conditions, ancilla_qubits
        )
        add_grover_diffusion(qc, var_qubits)

    qc.measure(var_qubits, cbits)

    return qc, var_qubits, cbits


def create_circuit_from_matrix(initial_state_matrix):
    state_array = np.array(initial_state_matrix)

    size = state_array.shape[0]
    initial_state = state_array.flatten()

    if size > 8 or size < 2:
        raise ValueError("Size of the board cannot be less than 2 or bigger than 8")

    # https://oeis.org/A075462
    num_solutions = [
        1,
        1,
        1,
        16,
        4,
        1,
        1,
        1,
    ][size - 1]

    # Optimal number of Grover iterations
    num_iterations = math.floor(math.pi / 4 * math.sqrt(2 ** (size**2) / num_solutions))

    qc, var_qubits, cbits = create_lights_out_circuit(
        size, initial_state, num_iterations
    )

    total_qubits = qc.num_qubits
    num_gates = qc.size()

    print(f"Grid size: {size}x{size}")
    print(f"Total qubits required: {total_qubits}")
    print(f"Total gates in circuit: {num_gates}")
    print(f"Grover iterations: {num_iterations}")

    return qc, var_qubits, cbits, size


def run_on_perfect_simulator(qc, shots=10_000):
    sampler = StatevectorSampler()
    result = sampler.run([qc], shots=shots).result()
    counts = result[0].data.cbits.get_counts()

    return counts


def run_on_ibm_simulator(qc, shots=10_000):
    backend = FakeMarrakesh()

    transpiled = transpile(qc, backend=backend, optimization_level=3)
    print("Transpiled")
    job = backend.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()

    return counts


def run_on_ibm_hardware(qc, shots=5_000, min_qubits=None):
    service = QiskitRuntimeService()

    # backend = service.least_busy(
    #     simulator=False, operational=True, min_num_qubits=min_qubits
    # )
    backend = service.backend("ibm_marrakesh")
    print(f"Will run on {backend.name}")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_circuit = pm.run(qc)

    target = backend.target

    # https://quantum.cloud.ibm.com/docs/en/guides/dynamical-decoupling-pass-manager
    # Improves performance on hardware by using dynamical decoupling
    X = XGate()

    dd_sequence = [X, X]

    dd_pm = PassManager(
        [
            ALAPScheduleAnalysis(target=target),
            PadDynamicalDecoupling(dd_sequence=dd_sequence, target=target),
        ]
    )

    isa_circuit = dd_pm.run(isa_circuit)
    print("Applied Dynamical Decoupling")

    # Big circuits might take too long to be drawn/saved
    isa_circuit.draw(output="mpl", idle_wires=False).savefig("ibm_circuit")

    num_gates = isa_circuit.size()
    print(f"Total gates in IBM circuit: {num_gates}")

    sampler = Sampler(backend)
    sampler.options.default_shots = shots

    print("Running the IBM sampler...")

    result = sampler.run([isa_circuit]).result()
    counts = result[0].data.cbits.get_counts()

    return counts


def plot_top_results(counts, top_n=20, filename="plot"):
    total_shots = sum(counts.values())
    if total_shots == 0:
        raise ValueError("Counts are empty (total shots = 0); nothing to plot.")

    top_n = min(top_n, len(counts))

    sorted_counts = dict(
        sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    )

    print(f"Top {top_n} results out of {len(counts)} total outcomes:")
    for state, count in sorted_counts.items():
        pct = 100.0 * (count / total_shots)
        print(f"  {state}: {count} ({pct:.2f}%)")

    fig = plot_histogram(sorted_counts, figsize=(12, 6), bar_labels=False)
    ax = fig.axes[0]

    label_map = {
        state: f"{count}\n{(100.0 * count / total_shots):.2f}%"
        for state, count in sorted_counts.items()
    }

    tick_positions = np.array(ax.get_xticks(), dtype=float)
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    labels_ordered = [label_map.get(txt, "") for txt in tick_texts]

    for bar in ax.patches:
        center_x = bar.get_x() + (bar.get_width() / 2)
        if tick_positions.size:
            idx = int(np.argmin(np.abs(tick_positions - center_x)))
        else:
            idx = 0

        label = labels_ordered[idx] if idx < len(labels_ordered) else ""
        if not label:
            continue

        height = bar.get_height()
        ax.annotate(
            label,
            (center_x, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

    if ax.patches:
        ax.set_ylim(0, max(p.get_height() for p in ax.patches) * 1.15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")


# 0 - turned off, 1 - turned on
# initial_state_matrix = [
#     [1, 1, 0],
#     [1, 0, 1],
#     [1, 0, 1],
# ]
# Solution: 101 011 100

initial_state_matrix = [
    [1, 0],
    [0, 1],
]
# Solution: 10 01

qc, var_qubits, cbits, size = create_circuit_from_matrix(initial_state_matrix)

qc.draw("mpl").savefig("circuit")

# counts = run_on_ibm_simulator(qc, shots=10_000)
# counts = run_on_ibm_hardware(qc, shots=100_000)
counts = run_on_perfect_simulator(qc, shots=100_000)

plot_top_results(counts, top_n=20, filename="plot")
