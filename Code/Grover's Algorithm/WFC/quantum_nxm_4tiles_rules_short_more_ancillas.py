from itertools import chain
from random import getrandbits
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import matplotlib.pyplot as plt

GRID_ROW_COUNT = 4
GRID_COL_COUNT = 4
GRID_TILE_COUNT = GRID_ROW_COUNT * GRID_COL_COUNT
GRID_EDGES = []
for row in range(GRID_ROW_COUNT):
    for col in range(GRID_COL_COUNT - 1):
        a = GRID_COL_COUNT * row + col
        b = GRID_COL_COUNT * row + col + 1
        GRID_EDGES.append((a, b))
        GRID_EDGES.append((b, a))
for row in range(GRID_ROW_COUNT - 1):
    for col in range(GRID_COL_COUNT):
        a = GRID_COL_COUNT * row + col
        b = GRID_COL_COUNT * (row + 1) + col
        GRID_EDGES.append((a, b))
        GRID_EDGES.append((b, a))

CONSTRAINT_COUNT = len(GRID_EDGES)
        
DATA_QUBIT_COUNT = GRID_TILE_COUNT * 2
ANCILLA_QUBIT_COUNT = CONSTRAINT_COUNT * 2 - 1
TOTAL_QUBIT_COUNT = DATA_QUBIT_COUNT + ANCILLA_QUBIT_COUNT
    
DATA_QUBITS = list(range(DATA_QUBIT_COUNT))
ANCILLA_QUBITS = list(range(DATA_QUBIT_COUNT, DATA_QUBIT_COUNT + ANCILLA_QUBIT_COUNT))

GROVER_ITERATIONS = 9
SHOT_COUNT = 1024

TILE_MAP = {"00": "Water", "01": "Sand", "10": "Grass", "11": "Jungle"}
COLOR_MAP = {"Water": (0.2, 0.4, 1.0), "Sand": (0.9, 0.85, 0.6), "Grass": (0.2, 0.7, 0.2), "Jungle": (0.0, 0.4, 0.0)}
VALID = {"Water": {"Water", "Sand"}, "Sand": {"Water", "Sand", "Grass"}, "Grass": {"Sand", "Grass", "Jungle"}, "Jungle": {"Grass", "Jungle"}}

def is_valid_grid(bitstring):
    tiles = [bitstring[i:i+2] for i in range(0, len(bitstring), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    for a, b in GRID_EDGES:
        if decoded[b] not in VALID[decoded[a]]:
            return False
    return True

def checkerboard_x(qc):
    for row in range(GRID_ROW_COUNT):
        for col in range(GRID_COL_COUNT):
            if (row + col) % 2 == 1:
                i = GRID_COL_COUNT * row + col
                data = DATA_QUBITS[i*2:i*2+2]
                qc.x(data)

def constraints():
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT)
    for i, (a, b) in enumerate(GRID_EDGES):
        data = list(chain(DATA_QUBITS[a*2:a*2+2], DATA_QUBITS[b*2+1:b*2+2]))
        ancilla = ANCILLA_QUBITS[i]
        qc.mcx(data, ancilla)
        qc.x(data)
        qc.mcx(data, ancilla)
        qc.x(data)
        qc.x(ancilla)
    i = 0
    j = CONSTRAINT_COUNT
    while j < ANCILLA_QUBIT_COUNT:
        qc.ccx(ANCILLA_QUBITS[i], ANCILLA_QUBITS[i+1], ANCILLA_QUBITS[j])
        i += 2
        j += 1
    return qc

def grover_oracle():
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT)
    constraints_qc = constraints()
    qc.compose(constraints_qc, inplace=True)
    qc.z(ANCILLA_QUBITS[-1])
    qc.compose(constraints_qc.inverse(), inplace=True)
    return qc

def estimate_optimal_iterations(valid_probability):
    p0 = 0
    for i in range(SHOT_COUNT):
        if is_valid_grid(f"{getrandbits(DATA_QUBIT_COUNT):0{DATA_QUBIT_COUNT}b}"):
            p0 += 1
    p0 /= SHOT_COUNT

    a = math.acos(1 - 2 * p0)
    b = math.acos(1 - 2 * valid_probability)
    c = GROVER_ITERATIONS * math.pi * (1 - a / math.pi)

    esitmate_up = c / (b - a)
    estimate_down = c / (math.tau - b - a)

    return esitmate_up, estimate_down

if __name__ == "__main__":
    # ONE-TIME SETUP: Uncomment and run once to save your token
    # QiskitRuntimeService.save_account(
    #     channel="ibm_quantum_platform",
    #     token="YOUR_TOKEN_HERE",
    #     overwrite=True
    # )
    # print("Token saved! Comment out this section and run again.")
    # exit()
    
    print("=" * 60)
    print("QUANTUM WAVE FUNCTION COLLAPSE - IBM QUANTUM HARDWARE")
    print("=" * 60)
    
    # Load IBM Quantum service
    print("\n[1/7] Connecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        print("✓ Connected successfully!")
    except Exception as e:
        print(f"✗ Error connecting: {e}")
        print("\nMake sure you've saved your token first!")
        print("Uncomment the save_account section above and add your token.")
        exit(1)
    
    # List available backends
    print("\n[2/7] Available quantum backends:")
    backends = service.backends(operational=True, simulator=False)
    for backend in backends:
        status = backend.status()
        queue = status.pending_jobs
        print(f"  • {backend.name}: {backend.num_qubits} qubits, Queue: {queue} jobs")
    
    # Select backend
    print(f"\n[3/7] Selecting backend (need {TOTAL_QUBIT_COUNT} qubits)...")
    try:
        backend = service.least_busy(
            operational=True, 
            simulator=False, 
            min_num_qubits=TOTAL_QUBIT_COUNT
        )
        print(f"✓ Selected: {backend.name} ({backend.num_qubits} qubits)")
        print(f"  Queue: {backend.status().pending_jobs} jobs waiting")
    except Exception as e:
        print(f"✗ No available backend with {TOTAL_QUBIT_COUNT} qubits")
        print(f"  Error: {e}")
        print("\n  Tip: Try reducing GRID_ROW_COUNT and GRID_COL_COUNT")
        exit(1)
    
    print(f"\nGrid configuration:")
    print(f"  Size: {GRID_ROW_COUNT}×{GRID_COL_COUNT}")
    print(f"  Data qubits: {DATA_QUBIT_COUNT}")
    print(f"  Ancilla qubits: {ANCILLA_QUBIT_COUNT}")
    print(f"  Total qubits: {TOTAL_QUBIT_COUNT}")
    print(f"  Grover iterations: {GROVER_ITERATIONS}")
    
    # Build circuit
    print("\n[4/7] Building quantum circuit...")
    grover_op = grover_operator(grover_oracle(), reflection_qubits=DATA_QUBITS)
    
    qc = QuantumCircuit(TOTAL_QUBIT_COUNT, DATA_QUBIT_COUNT)
    qc.h(DATA_QUBITS)
    qc.compose(grover_op.power(GROVER_ITERATIONS), inplace=True)
    checkerboard_x(qc)
    qc.measure(DATA_QUBITS, range(DATA_QUBIT_COUNT))
    
    # Get classical register name for later
    classical_register_name = qc.cregs[0].name
    print(f"✓ Circuit built: {qc.depth()} gates deep, {qc.size()} total gates")
    print(f"  Classical register name: '{classical_register_name}'")
    
    # Transpile
    print("\n[5/7] Transpiling for hardware...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    circuit_isa = pm.run(qc)
    print(f"✓ Transpiled: {circuit_isa.depth()} gates deep (optimized)")
    
    # Submit job
    print(f"\n[6/7] Submitting job to {backend.name}...")
    print(f"  Shots: {SHOT_COUNT}")
    sampler = Sampler(backend)
    job = sampler.run([circuit_isa], shots=SHOT_COUNT)
    print(f"✓ Job submitted!")
    print(f"  Job ID: {job.job_id()}")
    print(f"\n  Waiting for results...")
    print(f"  (This may take several minutes depending on queue)")
    print(f"  You can check status at: https://quantum.ibm.com/jobs/{job.job_id()}")
    
    # Get results
    try:
        result = job.result()
        # Access counts using the classical register name
        # result[0].data.<classical_register_name>.get_counts()
        pub_result = result[0]
        counts = getattr(pub_result.data, classical_register_name).get_counts()
        print(f"✓ Results received!")
        print(f"  Total measurements: {sum(counts.values())}")
        print(f"  Unique outcomes: {len(counts)}")
    except Exception as e:
        print(f"✗ Job failed: {e}")
        print(f"\nDebug info:")
        print(f"  Classical register name: '{classical_register_name}'")
        if 'result' in locals():
            print(f"  Result type: {type(result)}")
            if len(result) > 0:
                print(f"  Available data attributes: {list(result[0].data.__dict__.keys())}")
        exit(1)
    
    # Analyze results
    print("\n[7/7] Analyzing results...")
    valid_counts = {k: v for k, v in counts.items() if is_valid_grid(k)}
    total_valid = sum(valid_counts.values())
    
    valid_probability = total_valid / SHOT_COUNT
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Valid solutions: {total_valid}/{SHOT_COUNT} ({valid_probability*100:.1f}%)")
    print(f"Unique valid configurations: {len(valid_counts)}")
    
    if valid_counts:
        # Use valid solutions
        best = max(valid_counts, key=valid_counts.get)
        using_valid = True
    else:
        # No valid solutions found - use all results
        print("\n⚠ WARNING: No valid solutions found!")
        print("  This is likely due to hardware noise or incorrect Grover iterations.")
        print("  Selecting most frequent result from all measurements...")
        best = max(counts, key=counts.get)
        using_valid = False
    tiles = [best[i:i+2] for i in range(0, len(best), 2)]
    decoded = [TILE_MAP[t] for t in tiles]
    
    is_best_valid = is_valid_grid(best)
    best_count = valid_counts[best] if using_valid else counts[best]
    
    print(f"\nBest solution (appeared {best_count} times):")
    print(f"  {decoded}")
    print(f"  Valid: {'✓ YES' if is_best_valid else '✗ NO'}")
    if not using_valid:
        print(f"  ⚠ Note: Selected from ALL results (no valid solutions found)")
    
    if total_valid > 0:
        esitmate_up, estimate_down = estimate_optimal_iterations(valid_probability)
        print("\nOptimal Grover iteration estimates:")
        print(f"  If optimum > {GROVER_ITERATIONS}: {esitmate_up:.2f} iterations")
        print(f"  If optimum < {GROVER_ITERATIONS}: {estimate_down:.2f} iterations")
    else:
        print("\n⚠ Cannot estimate optimal iterations (no valid solutions)")
    
    # Visualize
    print("\n[8/7] Generating visualization...")
    grid = np.zeros((GRID_ROW_COUNT, GRID_COL_COUNT, 3))
    for i, tile in enumerate(decoded):
        grid[i // GRID_COL_COUNT, i % GRID_COL_COUNT] = COLOR_MAP[tile]
    
    fig, ax = plt.subplots(figsize=(max(6, GRID_COL_COUNT*2), max(6, GRID_ROW_COUNT*2)))
    ax.imshow(grid, interpolation='nearest')
    ax.axis("off")
    
    for i, tile in enumerate(decoded):
        r, c = i // GRID_COL_COUNT, i % GRID_COL_COUNT
        ax.text(c, r, f"{i}\n{tile}", ha='center', va='center', fontsize=max(8, 12-max(GRID_ROW_COUNT,GRID_COL_COUNT)), 
                color='white', weight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    for i in range(GRID_ROW_COUNT + 1):
        ax.axhline(i - 0.5, color='black', linewidth=2)
    for j in range(GRID_COL_COUNT + 1):
        ax.axvline(j - 0.5, color='black', linewidth=2)
    
    title_color = 'green' if is_best_valid else 'red'
    if GRID_ROW_COUNT <= 3 and GRID_COL_COUNT <= 3:
        grid_str = '\n'.join([str(decoded[row*GRID_COL_COUNT:(row+1)*GRID_COL_COUNT]) for row in range(GRID_ROW_COUNT)])
        title = f"{GRID_ROW_COUNT}×{GRID_COL_COUNT} WFC Grid - IBM Quantum Hardware\n{backend.name}\n{grid_str}"
        ax.set_title(title, fontsize=11, color=title_color, weight='bold')
    else:
        ax.set_title(f"{GRID_ROW_COUNT}×{GRID_COL_COUNT} WFC Grid - {backend.name}", fontsize=14, color=title_color, weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'=' * 60}")
    print("✓ Complete!")
    print(f"{'=' * 60}")