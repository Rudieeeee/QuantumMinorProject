"""
Quantum Grover's algorithm solver for 2x2 Sudoku/Latin-square.

A 2x2 Sudoku puzzle requires:
- Each row contains 1 and 2
- Each column contains 1 and 2

Encoding per cell: 1 bit represents values 1 and 2
- 0 -> 1
- 1 -> 2

Total: k empty cells → k qubits (where k ≤ 4)
"""
from typing import List, Optional, Tuple
import math
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

Grid = List[List[int]]


def print_grid(g: Grid) -> None:
    """Print a 2x2 grid in readable format."""
    for row in g:
        print(" ".join(str(x) if x != 0 else "." for x in row))


def enumerate_valid_assignments_2x2(
    puzzle: Grid, empty_positions: List[Tuple[int, int]]
) -> List[str]:
    """Enumerate all valid 2x2 Sudoku solutions for given empty positions.
    
    For each empty position, we try values 1 or 2 (encoded as 0 or 1).
    A valid assignment must satisfy:
    - Each row contains both 1 and 2
    - Each column contains both 1 and 2
    
    Args:
        puzzle: The puzzle grid with 0s for empty cells
        empty_positions: List of (row, col) tuples for empty cells
    
    Returns:
        List of bitstrings where bit i encodes cell i (0->1, 1->2)
    """
    k = len(empty_positions)
    if k == 0:
        return []
    
    solutions = []
    
    # Iterate over all 2^k assignments (each empty cell can be 1 or 2)
    for assignment_idx in range(2 ** k):
        grid = [row[:] for row in puzzle]
        bitstring_parts = []
        
        # Assign values to empty positions based on bits
        for bit_pos in range(k):
            bit = (assignment_idx >> bit_pos) & 1
            val = 2 if bit == 1 else 1  # 0->1, 1->2
            r, c = empty_positions[bit_pos]
            grid[r][c] = val
            bitstring_parts.append(str(bit))
        
        # Validate the completed grid
        valid_grid = True
        
        # Check rows: each row must have both 1 and 2
        for i in range(2):
            if set(grid[i]) != {1, 2}:
                valid_grid = False
                break
        
        # Check columns: each column must have both 1 and 2
        if valid_grid:
            for j in range(2):
                if {grid[0][j], grid[1][j]} != {1, 2}:
                    valid_grid = False
                    break
        
        if valid_grid:
            # Build bitstring: bit i is at position i in the string
            bitstring = "".join(bitstring_parts)
            solutions.append(bitstring)
    
    return solutions



def build_grover_circuit_2x2(
    empty_positions: List[Tuple[int, int]], solutions: List[str]
):
    """Build a Grover circuit for 2x2 Sudoku.
    
    Uses 1 bit per cell. Returns (circuit, phase_ancilla_qubit_index, num_data_qubits).
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit is required for Grover solver")
    
    k = len(empty_positions)
    n = k  # 1 bit per cell
    
    # Ancilla qubits: at least (n-2) for multi-controlled X, plus 1 for phase
    anc_for_mcx = max(0, n - 2)
    total_qubits = n + anc_for_mcx + 1
    
    qc = QuantumCircuit(total_qubits, n)
    
    data_qubits = list(range(n))
    anc_qubits = list(range(n, n + anc_for_mcx)) if anc_for_mcx > 0 else []
    phase_anc = n + anc_for_mcx
    
    # Initial superposition
    qc.h(data_qubits)
    
    # Define oracle: mark each solution with a phase flip
    def apply_solution_phase(bs: str):
        """Apply phase flip for a specific solution bitstring.
        
        bs: bitstring where bs[i] is the bit for qubit i
        """
        # Apply X gates to qubits where bit should be 0 (to invert for AND)
        for i in range(len(bs)):
            if bs[i] == "0":
                qc.x(data_qubits[i])
        
        # Prepare phase ancilla in |-⟩
        qc.x(phase_anc)
        qc.h(phase_anc)
        
        # Multi-controlled X: if all data qubits are 1, flip phase ancilla
        if len(data_qubits) == 0:
            # No data qubits, just apply X
            qc.x(phase_anc)
        elif len(data_qubits) == 1:
            qc.cx(data_qubits[0], phase_anc)
        elif len(data_qubits) == 2:
            qc.ccx(data_qubits[0], data_qubits[1], phase_anc)
        else:
            # Use mcx with ancilla
            qc.mcx(data_qubits, phase_anc, anc_qubits)
        
        # Unprepare phase ancilla
        qc.h(phase_anc)
        qc.x(phase_anc)
        
        # Undo X gates
        for i in range(len(bs)):
            if bs[i] == "0":
                qc.x(data_qubits[i])
    
    # Oracle: apply phase flip for each solution
    for bs in solutions:
        apply_solution_phase(bs)
    
    # Diffusion operator (inversion about average)
    qc.h(data_qubits)
    qc.x(data_qubits)
    qc.x(phase_anc)
    qc.h(phase_anc)
    
    if len(data_qubits) == 0:
        qc.x(phase_anc)
    elif len(data_qubits) == 1:
        qc.cx(data_qubits[0], phase_anc)
    elif len(data_qubits) == 2:
        qc.ccx(data_qubits[0], data_qubits[1], phase_anc)
    else:
        qc.mcx(data_qubits, phase_anc, anc_qubits)
    
    qc.h(phase_anc)
    qc.x(phase_anc)
    qc.x(data_qubits)
    qc.h(data_qubits)
    
    # Measure data qubits
    qc.measure(range(n), range(n))
    
    return qc, phase_anc, n



def grover_solve_2x2(puzzle: Grid) -> Optional[Grid]:
    """Solve a 2x2 Sudoku puzzle using Grover's algorithm.
    
    Returns the solved grid if Qiskit is available, otherwise None.
    """
    try:
        from qiskit import transpile, QuantumCircuit
    except ImportError:
        print("Qiskit not installed; cannot run Grover solver.")
        return None
    
    # Get simulator backend
    AerSimulator = None
    BasicAerSimulator = None
    try:
        from qiskit_aer import AerSimulator as _AerSim
        AerSimulator = _AerSim
    except ImportError:
        try:
            from qiskit.providers.aer import AerSimulator as _AerSim
            AerSimulator = _AerSim
        except ImportError:
            try:
                from qiskit.providers.basicaer import QasmSimulator as _Basic
                BasicAerSimulator = _Basic
            except ImportError:
                print("Qiskit Aer and BasicAer not available.")
                return None
    
    # Find empty positions
    empty_positions = [
        (i, j) for i in range(2) for j in range(2) if puzzle[i][j] == 0
    ]
    
    if len(empty_positions) == 0:
        print("No empty cells; puzzle already solved.")
        return [row[:] for row in puzzle]
    
    k = len(empty_positions)
    print(f"Running Grover over {k} empty cell(s) (search space 2^{k} = {2**k})")
    
    # Enumerate valid assignments
    solutions = enumerate_valid_assignments_2x2(puzzle, empty_positions)
    if not solutions:
        print("No valid assignments found.")
        return None
    
    print(f"Found {len(solutions)} valid assignment(s).")
    
    # Build Grover circuit
    qc, phase_anc, n = build_grover_circuit_2x2(empty_positions, solutions)
    
    # Calculate number of iterations
    N = 2 ** k
    M = max(1, len(solutions))
    iters = max(1, int(math.floor((math.pi / 4) * math.sqrt(N / M))))
    print(f"Using {iters} Grover iteration(s)")
    
    # Rebuild circuit with iterations
    qc = QuantumCircuit(n + max(0, n - 2) + 1, n)
    data_qubits = list(range(n))
    anc_for_mcx = max(0, n - 2)
    anc_qubits = list(range(n, n + anc_for_mcx)) if anc_for_mcx > 0 else []
    phase_anc = n + anc_for_mcx
    
    # Initial superposition
    qc.h(data_qubits)
    
    def apply_solution_phase_local(bs: str):
        for i in range(len(bs)):
            if bs[i] == "0":
                qc.x(data_qubits[i])
        qc.x(phase_anc)
        qc.h(phase_anc)
        if len(data_qubits) == 0:
            qc.x(phase_anc)
        elif len(data_qubits) == 1:
            qc.cx(data_qubits[0], phase_anc)
        elif len(data_qubits) == 2:
            qc.ccx(data_qubits[0], data_qubits[1], phase_anc)
        else:
            qc.mcx(data_qubits, phase_anc, anc_qubits)
        qc.h(phase_anc)
        qc.x(phase_anc)
        for i in range(len(bs)):
            if bs[i] == "0":
                qc.x(data_qubits[i])
    
    def oracle_local():
        for bs in solutions:
            apply_solution_phase_local(bs)
    
    def diffusion_local():
        qc.h(data_qubits)
        qc.x(data_qubits)
        qc.x(phase_anc)
        qc.h(phase_anc)
        if len(data_qubits) == 0:
            qc.x(phase_anc)
        elif len(data_qubits) == 1:
            qc.cx(data_qubits[0], phase_anc)
        elif len(data_qubits) == 2:
            qc.ccx(data_qubits[0], data_qubits[1], phase_anc)
        else:
            qc.mcx(data_qubits, phase_anc, anc_qubits)
        qc.h(phase_anc)
        qc.x(phase_anc)
        qc.x(data_qubits)
        qc.h(data_qubits)
    
    # Apply Grover iterations
    for _ in range(iters):
        oracle_local()
        diffusion_local()
    
    # Measure
    qc.measure(range(n), range(n))
    
    # Choose backend and transpile
    if AerSimulator is not None:
        backend = AerSimulator()
        qc_t = transpile(qc, backend=backend, basis_gates=["u3", "cx"])
    elif BasicAerSimulator is not None:
        backend = BasicAerSimulator()
        qc_t = transpile(qc, backend=backend)
    else:
        print("No simulator backend available.")
        return None
    
    print("\nGrover circuit (transpiled):")
    print(qc_t)
    
    # Execute
    job = backend.run(qc_t, shots=2048)
    res = job.result()
    counts = res.get_counts()
    print("\nMeasurement counts:")
    print(counts)
    
    # Decode best result
    best = max(counts, key=counts.get)
    # Qiskit returns bitstring with qubit 0 at rightmost (little-endian)
    # We want to reverse to get our expected order
    best_be = best[::-1]
    
    print(f"Best measurement (binary): {best} -> reversed: {best_be}")
    
    vals = []
    for i in range(k):
        bit = int(best_be[i])
        val = 2 if bit == 1 else 1  # 0->1, 1->2
        vals.append(val)
    
    # Fill in the solution
    grid = [row[:] for row in puzzle]
    for i, (r, c) in enumerate(empty_positions):
        grid[r][c] = vals[i]
    
    print("\nDecoded best assignment:")
    print_grid(grid)
    
    return grid

def grover_solve_ibm(puzzle: Grid) -> Optional[Grid]:
    """
    Solve 2x2 Sudoku using Grover's algorithm on IBM Quantum Hardware.
    """
    print("Connecting to IBM Quantum Service...")
    try:
        service = QiskitRuntimeService(channel="ibm_quantum_platform")
    except Exception as e:
        print(f"Connection with 'ibm_quantum_platform' failed: {e}")
        try:
            service = QiskitRuntimeService()
        except Exception as e2:
            print(f"{e2}")
            return None

    empty_positions = [(i, j) for i in range(2) for j in range(2) if puzzle[i][j] == 0]
    k = len(empty_positions)
    
    if k == 0:
        print("Puzzle already solved.")
        return [row[:] for row in puzzle]

    num_qubits_needed = k + 1 + max(0, k - 2)
    print(f"Requesting backend with at least {num_qubits_needed} qubits...")
    
    try:
        backend = service.least_busy(
            operational=True, 
            simulator=False, 
            min_num_qubits=num_qubits_needed
        )
        print(f"Selected backend: {backend.name}")
    except Exception as e:
        print("No suitable backend found. Try again later or check your access.")
        return None

    solutions = enumerate_valid_assignments_2x2(puzzle, empty_positions)
    if not solutions:
        print("No valid assignments exist.")
        return None
    
    print(f"Found {len(solutions)} solution(s). Constructing circuit...")

    N = 2 ** k
    M = len(solutions)
    iters = max(1, int(math.floor((math.pi / 4) * math.sqrt(N / M))))
    print(f"Grover iterations: {iters}")

    # circuit setup
    n = k
    anc_for_mcx = max(0, n - 2)
    total_qubits = n + anc_for_mcx + 1
    
    qc = QuantumCircuit(total_qubits, n)
    data_qubits = list(range(n))
    anc_qubits = list(range(n, n + anc_for_mcx)) if anc_for_mcx > 0 else []
    phase_anc = n + anc_for_mcx

    # initialization
    qc.h(data_qubits)
    
    # oracle Helper
    def apply_oracle(target_solutions):
        for bs in target_solutions:
            for i in range(len(bs)):
                if bs[i] == "0":
                    qc.x(data_qubits[i])
            qc.x(phase_anc)
            qc.h(phase_anc)
            if n == 1: qc.cx(data_qubits[0], phase_anc)
            elif n == 2: qc.ccx(data_qubits[0], data_qubits[1], phase_anc)
            else: qc.mcx(data_qubits, phase_anc, anc_qubits)
            qc.h(phase_anc)
            qc.x(phase_anc)
            for i in range(len(bs)):
                if bs[i] == "0":
                    qc.x(data_qubits[i])

    # diffusion Helper
    def apply_diffusion():
        qc.h(data_qubits)
        qc.x(data_qubits)
        qc.x(phase_anc)
        qc.h(phase_anc)
        if n == 1: qc.cx(data_qubits[0], phase_anc)
        elif n == 2: qc.ccx(data_qubits[0], data_qubits[1], phase_anc)
        else: qc.mcx(data_qubits, phase_anc, anc_qubits)
        qc.h(phase_anc)
        qc.x(phase_anc)
        qc.x(data_qubits)
        qc.h(data_qubits)

    # apply Grover Iterations
    for _ in range(iters):
        apply_oracle(solutions)
        apply_diffusion()

    qc.measure(data_qubits, range(n))

    # transpile and Execute
    print("Transpiling circuit...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(qc)

    print("Submitting job to IBM Quantum...")
    sampler = Sampler(mode=backend)
    total_shots = 4096
    job = sampler.run([isa_circuit], shots=total_shots)
    print(f"Job ID: {job.job_id()}")
    
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()
    print("\nHardware Counts:", counts)

    best_bitstring = max(counts, key=counts.get)
    best_be = best_bitstring[::-1] # reverse for little-endian
    
    print(f"Most frequent measurement: {best_bitstring} (interpreted as {best_be})")

    vals = []
    for i in range(k):
        if i < len(best_be):
            bit = int(best_be[i])
            vals.append(2 if bit == 1 else 1)
        else:
            vals.append(1)
    
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_result = sorted_counts[0][0]

    print(f"\nTop Result from Hardware: {top_result}")

    top_20 = sorted_counts[:20]
    labels = [x[0] for x in top_20]
    probabilities = [x[1] / total_shots for x in top_20]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, probabilities, color='indigo')
    for bar in bars:
        yval = bar.get_height()
        
        if yval > 0.01:
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                yval + 0.01,              
                f'{yval:.2%}',             
                ha='center', va='bottom', 
                fontsize=14, rotation=0,
                fontweight = 'bold'    
            )

    plt.ylim(0, max(probabilities) * 1.3)
    plt.xlabel('Measurement Bitstring')
    plt.ylabel('Probability')
    plt.title(f'Real Hardware Results ({backend.name})\nShots={total_shots}')
    plt.xticks(rotation=90, fontname='Monospace')
    plt.tight_layout()
    plt.savefig('ibm_hardware_result_2x2.png')
    print("Saved hardware plot to 'ibm_hardware_result_2x2.png'")

    final_grid = [row[:] for row in puzzle]
    for i, (r, c) in enumerate(empty_positions):
        final_grid[r][c] = vals[i]

    print("\nFinal Solution from Quantum Hardware:")
    print_grid(final_grid)
    return final_grid

if __name__ == "__main__":
    puzzle = [
        [0, 0],
        [2, 1]
    ]

    print("Input Puzzle:")
    print_grid(puzzle)
    print("-" * 20)

    grover_solve_ibm(puzzle)