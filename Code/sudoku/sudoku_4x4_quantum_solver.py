from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import math
from sudoku_4x4_constraints import generate_constraints

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

value_constraints = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2), (5, 1), (5, 2)]
relative_constraints = [(0, 1), (0, 2), (0, 5)]

# value_constraints, relative_constraints = generate_constraints(18)

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

# Building the circuit
var = QuantumRegister(16, name='v')
scratch = QuantumRegister(2, name='s')
block_ancillas = QuantumRegister(4, name='blk')
group_results = QuantumRegister(5, name='grp')
out = QuantumRegister(1, name='out')
c = ClassicalRegister(16, name='c')

qc = QuantumCircuit(var, scratch, block_ancillas, group_results, out, c)

qc.h(var)
qc.x(out); qc.h(out)

num_iterations = 200

for i in range(num_iterations):
    oracle(qc, var, scratch, block_ancillas, group_results, out)

    qc.h(var)
    qc.x(var)
    qc.h(var[-1])
    qc.mcx(var[:-1], var[-1])
    qc.h(var[-1])
    qc.x(var)
    qc.h(var)

qc.measure(var, c)

print(f"Circuit Width: {qc.num_qubits} qubits (Should be < 30)")


# Run with simulator

backend = AerSimulator(method='matrix_product_state')

print("Starting Simulation...\n")
job = backend.run(transpile(qc, backend, optimization_level=2), shots=5000)
result = job.result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
top_bitstring = sorted_counts[0][0]

print(f"Most probable bitstring: {top_bitstring}")

print("SOLUTION")
bits = top_bitstring[::-1]
for i in range(8):
    chunk = bits[i * 2 : i * 2 + 2]
    val = int(chunk[::-1], 2) + 1
    print(f"Unknown x{i}: {val}")

