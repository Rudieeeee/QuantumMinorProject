from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import math
from sudoku_4x4_constraints import generate_constraints
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel, depolarizing_error

# value_constraints = [
#     (0, 0), (0, 2), (0, 3),
#     (1, 1), (1, 2), (1, 3),
#     (2, 0), (2, 1), (2, 2),
#     (3, 0), (3, 1), (3, 3),
#     (4, 1), (4, 2), (4, 3),
#     (5, 0), (5, 2), (5, 3),
#     (6, 0), (6, 1), (6, 3),
#     (7, 0), (7, 1), (7, 2),
# ]
# relative_constraints = [
#     (0, 1),
# ]

# value_constraints = [(7, 3), (2, 0), (5, 3), (1, 3), (3, 2), (5, 0), (6, 2), (7, 2)]
# relative_constraints = [(4, 6), (3, 4), (0, 1), (1, 2)]

nr_unkowns = 6
value_constraints, relative_constraints = generate_constraints(nr_unkowns, 28, test=True)

# Check a value contraint
def check_value_constraint(qc, var_q, value, ancilla):
    # Flips ancilla if var_qubits == 'value'
    binary_str = format(value, '02b')
    if binary_str[0] == '0': qc.x(var_q[1])
    if binary_str[1] == '0': qc.x(var_q[0])
    qc.ccx(var_q[0], var_q[1], ancilla)
    if binary_str[0] == '0': qc.x(var_q[1])
    if binary_str[1] == '0': qc.x(var_q[0])

# Check relative constraint
def check_relative_constraint(qc, qA, qB, scratch, result_ancilla):
    # Flips result_ancilla if qA != qB
    s0, s1 = scratch
    qc.cx(qA[0], s0); qc.cx(qB[0], s0)
    qc.cx(qA[1], s1); qc.cx(qB[1], s1)
    qc.x(s0); qc.x(s1)
    qc.ccx(s0, s1, result_ancilla)
    qc.x(result_ancilla)
    qc.x(s0); qc.x(s1)
    qc.cx(qB[1], s1); qc.cx(qA[1], s1)
    qc.cx(qB[0], s0); qc.cx(qA[0], s0)


# Defining the oracle
def oracle(qc, var, scratch, block_ancillas, group_results, out):

    # list of all contraints
    all_constraints = []
    for vc in value_constraints:
        all_constraints.append(('value', vc))
    for rc in relative_constraints:
        all_constraints.append(('relative', rc))

    # Batch processing
    # We use blocks of size 4 qubits for every batch of 4 constraints.
    block_size = len(block_ancillas)
    # We also need the number of blocks for knowing the size of the qubit register where
    # we are going to store the results group_results
    num_blocks = math.ceil(len(all_constraints) / block_size)

    for i in range(num_blocks):
        # current batch of constraints
        batch = all_constraints[i * block_size : (i + 1) * block_size]

        # compute each constraint in the batch
        for j, (tp, data) in enumerate(batch):
            target_ancilla = block_ancillas[j]
            if tp == 'value':
                idx, val = data
                check_value_constraint(qc, [var[idx * 2], var[idx * 2 + 1]], val, target_ancilla)
                qc.x(target_ancilla)
            else:
                idx_a, idx_b = data
                qA, qB = [var[idx_a * 2], var[idx_a * 2 + 1]], [var[idx_b * 2], var[idx_b * 2 + 1]]
                check_relative_constraint(qc, qA, qB, scratch, target_ancilla)
        
        # check if all elements in the block are valid and store result in group_results[i]
        qc.mcx(block_ancillas[:len(batch)], group_results[i])

        # uncompute the block ancillas for reuse in next block
        for j, (tp, data) in reversed(list(enumerate(batch))):
            target_ancilla = block_ancillas[j]
            if tp == 'value':
                idx, val = data
                qc.x(target_ancilla)
                check_value_constraint(qc, [var[idx * 2], var[idx * 2 + 1]], val, target_ancilla)
            else:
                idx_a, idx_b = data
                qA, qB = [var[idx_a * 2], var[idx_a * 2 + 1]], [var[idx_b * 2], var[idx_b * 2 + 1]]
                check_relative_constraint(qc, qA, qB, scratch, target_ancilla)

    # Phase kickback
    # check if ALL group results are 1
    qc.mcx(group_results[:num_blocks], out)

    # uncompute everything for reuse of qubits
    for i in reversed(range(num_blocks)):
        # current batch of constraints
        batch = all_constraints[i * block_size : (i + 1) * block_size]

        # uncompute each constraint in the batch
        for j, (tp, data) in enumerate(batch):
            target_ancilla = block_ancillas[j]
            if tp == 'value':
                idx, val = data
                check_value_constraint(qc, [var[idx * 2], var[idx * 2 + 1]], val, target_ancilla)
                qc.x(target_ancilla)
            else:
                idx_a, idx_b = data
                qA, qB = [var[idx_a * 2], var[idx_a * 2 + 1]], [var[idx_b * 2], var[idx_b * 2 + 1]]
                check_relative_constraint(qc, qA, qB, scratch, target_ancilla)
        
        # uncompute group result
        qc.mcx(block_ancillas[:len(batch)], group_results[i])

        # uncompute again
        for j, (tp, data) in reversed(list(enumerate(batch))):
            target_ancilla = block_ancillas[j]
            if tp == 'value':
                idx, val = data
                qc.x(target_ancilla)
                check_value_constraint(qc, [var[idx * 2], var[idx * 2 + 1]], val, target_ancilla)
            else:
                idx_a, idx_b = data
                qA, qB = [var[idx_a * 2], var[idx_a * 2 + 1]], [var[idx_b * 2], var[idx_b * 2 + 1]]
                check_relative_constraint(qc, qA, qB, scratch, target_ancilla)

def build_circuit(iterations) :
    block_size = 4
    actual_blocks = math.ceil((len(value_constraints) + len(relative_constraints)) / block_size)
    # Building the circuit
    var = QuantumRegister(12, name='v')
    scratch = QuantumRegister(2, name='s')
    block_ancillas = QuantumRegister(4, name='blk')
    group_results = QuantumRegister(actual_blocks, name='grp')
    out = QuantumRegister(1, name='out')
    c = ClassicalRegister(12, name='c')

    qc = QuantumCircuit(var, scratch, block_ancillas, group_results, out, c)

    qc.h(var)
    qc.x(out); qc.h(out)

    for i in range(iterations):
        oracle(qc, var, scratch, block_ancillas, group_results, out)

        qc.h(var)
        qc.x(var)
        qc.h(var[-1])
        qc.mcx(var[:-1], var[-1])
        qc.h(var[-1])
        qc.x(var)
        qc.h(var)

    qc.measure(var, c)
    return qc

# RUN ON AER SIMULATOR
def run_aer_simulation(num_iterations):
    qc_result = build_circuit(num_iterations)
    print(f"Circuit Width: {qc_result.num_qubits} qubits (Should be < 30)")
    print(f"Running simulation with {num_iterations} iterations...")
    backend = AerSimulator(method='statevector')
    
    total_shots = 2048
    transpiled_qc = transpile(qc_result, backend)
    job = backend.run(transpiled_qc, shots=total_shots)
    result = job.result()
    counts = result.get_counts()

    print(f"Original Gates: {qc_result.size()}")
    print(f"Transpiled Gates (Real Hardware): {transpiled_qc.size()}")
    print(f"Depth: {transpiled_qc.depth()}")

    # process Data for Plotting
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_counts[:20]
    labels = [x[0] for x in top_20]
    probabilities = [x[1] / total_shots for x in top_20]

    print("Top Result:", labels[0])

    # plotting
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, probabilities, color='skyblue')
    bars[0].set_color('crimson')
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

    plt.xlabel('Measurement Bitstring (Result)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Grover Search Results (Top 20 States)\nShots={total_shots}', fontsize=14)
    plt.xticks(rotation=90, fontname='Monospace')
    plt.tight_layout()
    plt.show()

    # decode
    bits = labels[0][::-1]
    print("\n--- DECODED SUDOKU ---")
    for i in range(nr_unkowns):
        chunk = bits[i*2 : i*2+2]
        val = int(chunk[::-1], 2) + 1
        print(f"Variable x{i}: {val}")


# RUN ON IBM QUANTUM

def run_ibm(num_iterations):
    print(f"\nIBM QUANTUM RUN")

    service = QiskitRuntimeService()

    print("Searching for least busy backend")
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=27)
    print(f"Selected Backend: {backend.name}")

    print("Transpiling circuit for hardware...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3) # optimization 3 to reduce depth of grover
    qc_result = build_circuit(num_iterations)
    isa_qc = pm.run(qc_result) 
    
    print(f"Original Gates: {qc_result.size()}")
    print(f"Transpiled Gates (Real Hardware): {isa_qc.size()}")
    print(f"Depth: {isa_qc.depth()}")

    print(f"Submitting job to {backend.name}")
    sampler = Sampler(mode=backend)

    # running the job
    total_shots = 4096
    job = sampler.run([isa_qc], shots=total_shots)

    print(f"Job submitted! ID: {job.job_id()}")
    print("Waiting for results...")

    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_result = sorted_counts[0][0]

    print(f"\nTop Result from Hardware: {top_result}")

    top_20 = sorted_counts[:20]
    labels = [x[0] for x in top_20]
    probabilities= [x[1] / total_shots for x in top_20]

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
    plt.savefig('ibm_hardware_result.png')
    print("Saved hardware plot to 'ibm_hardware_result.png'")

    bits = top_result[::-1]
    print("\n--- DECODED HARDWARE RESULT ---")

    for i in range(nr_unkowns):
        if (i*2 + 2) <= len(bits):
            chunk = bits[i*2 : i*2+2]
            val = int(chunk[::-1], 2) + 1
            print(f"Variable x{i}: {val}")

def run_ibm_death_curve(target_bitstring):
    print(f"\nIBM QUANTUM: COHERENCE LIMIT ANALYSIS")
    
    service = QiskitRuntimeService()
    print("Searching for least busy backend...")
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=27)
    print(f"Selected Backend: {backend.name}")

    iteration_steps = [1, 5, 10, 20, 30, 40, 48] 
    print(f"Building circuits for iterations: {iteration_steps}")
    
    raw_circuits = []
    for k in iteration_steps:
        qc = build_circuit(k)
        raw_circuits.append(qc)

    print("Transpiling batch...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuits = pm.run(raw_circuits) 

    print(f"Submitting batch job to {backend.name}...")
    sampler = Sampler(mode=backend)
    
    total_shots = 4096
    job = sampler.run(isa_circuits, shots=total_shots)
    
    print(f"Job submitted! ID: {job.job_id()}")
    print("Waiting for results...")
    result = job.result()
    print("Job complete! Processing data...")
    hw_probabilities = []
    
    for i, pub_result in enumerate(result):
        counts = pub_result.data.c.get_counts()
        total_counts = sum(counts.values())
    
        correct_count = counts.get(target_bitstring, 0)
        
        prob = correct_count / total_counts
        hw_probabilities.append(prob)
        print(f"  Iter {iteration_steps[i]}: {prob:.2%}")

    plt.figure(figsize=(10, 6))    
    plt.plot(iteration_steps, hw_probabilities, 'o-', color='purple', linewidth=2, label=f'IBM Hardware ({backend.name})')
    import numpy as np
    x_theory = np.linspace(0, 50, 100)
    theta = np.arcsin(1/np.sqrt(4096))
    y_theory = np.sin((2 * x_theory + 1) * theta)**2
    plt.plot(x_theory, y_theory, '--', color='gray', alpha=0.5, label='Ideal Theory')

    plt.xlabel('Grover Iterations', fontsize=12)
    plt.ylabel('Probability of Correct Solution', fontsize=12)
    plt.title(f'The Reality Gap: Ideal vs Real Hardware ({backend.name})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = 'ibm_death_curve.png'
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()


run_aer_simulation(50)
# run_ibm(45)
# run_ibm_death_curve("010110011000")