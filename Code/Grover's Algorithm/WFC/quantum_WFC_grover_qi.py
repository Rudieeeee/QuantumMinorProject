import random
from PIL import Image

from qiskit import QuantumCircuit
from qiskit_quantuminspire.qi_provider import QIProvider

# ======================================================
# Quantum Inspire setup
# ======================================================
provider = QIProvider()
backend = provider.get_backend("QX emulator")

# ======================================================
# TILE SET
# ======================================================
TILES = ["grass", "sand", "water", "forest", "rock"]

ADJACENCY = {
    "grass": {"grass", "sand", "forest"},
    "sand": {"grass", "sand", "water"},
    "water": {"sand", "water"},
    "forest": {"grass", "forest", "rock"},
    "rock": {"forest", "rock"},
}

COLORS = {
    "grass": (34, 139, 34),
    "sand": (194, 178, 128),
    "water": (30, 144, 255),
    "forest": (0, 100, 0),
    "rock": (110, 110, 110),
}

GRID_W, GRID_H = 2, 2
CELL_SIZE = 40
SHOTS = 256

# ======================================================
# Classical WFC helpers
# ======================================================
def new_grid():
    return [[set(TILES) for _ in range(GRID_W)] for _ in range(GRID_H)]


def neighbors(x, y):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny


def entropy(cell):
    return len(cell)

# ======================================================
# Toffoli decomposition (QI compatible)
# ======================================================
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

# ======================================================
# Grover oracle (mark a single index)
# ======================================================
def grover_oracle(circ, target_bits):
    for i, bit in enumerate(target_bits):
        if bit == "0":
            circ.x(i)

    circ.h(2)
    decomposed_toffoli(circ, 0, 1, 2)
    circ.h(2)

    for i, bit in enumerate(target_bits):
        if bit == "0":
            circ.x(i)

# ======================================================
# Grover diffusion
# ======================================================
def grover_diffusion(circ):
    circ.h([0, 1, 2])
    circ.x([0, 1, 2])

    circ.h(2)
    decomposed_toffoli(circ, 0, 1, 2)
    circ.h(2)

    circ.x([0, 1, 2])
    circ.h([0, 1, 2])

# ======================================================
# Grover-based tile choice (QI)
# ======================================================
def grover_choose_tile(grid, x, y):
    valid_tiles = set(TILES)

    for nx, ny in neighbors(x, y):
        if len(grid[ny][nx]) == 1:
            valid_tiles &= ADJACENCY[next(iter(grid[ny][nx]))]

    if not valid_tiles:
        return random.choice(list(grid[y][x]))

    # pick ONE valid tile to mark (Grover is single-solution here)
    tile = random.choice(list(valid_tiles))
    tile_index = TILES.index(tile)
    target_bits = format(tile_index, "03b")

    qc = QuantumCircuit(3, 3)

    # superposition
    qc.h([0, 1, 2])

    # one Grover iteration
    grover_oracle(qc, target_bits)
    grover_diffusion(qc)

    qc.measure([0, 1, 2], [0, 1, 2])

    job = backend.run(qc, shots=SHOTS)
    result = job.result()
    counts = result.get_counts()

    best = max(counts, key=counts.get)
    idx = int(best[::-1], 2)

    if idx < len(TILES):
        return TILES[idx]

    return tile

# ======================================================
# Propagation
# ======================================================
def propagate(grid, start):
    stack = [start]

    while stack:
        x, y = stack.pop()
        for nx, ny in neighbors(x, y):
            allowed = set()
            for t in grid[y][x]:
                allowed |= ADJACENCY[t]

            before = grid[ny][nx]
            after = before & allowed

            if not after:
                return False

            if after != before:
                grid[ny][nx] = after
                stack.append((nx, ny))

    return True

# ======================================================
# Collapse
# ======================================================
def collapse(grid):
    cells = [
        (x, y)
        for y in range(GRID_H)
        for x in range(GRID_W)
        if len(grid[y][x]) > 1
    ]

    if not cells:
        return True

    x, y = min(cells, key=lambda c: entropy(grid[c[1]][c[0]]))
    grid[y][x] = {grover_choose_tile(grid, x, y)}
    return propagate(grid, (x, y))

# ======================================================
# Run WFC
# ======================================================
def run_wfc():
    while True:
        grid = new_grid()
        ok = True
        while ok:
            if all(len(grid[y][x]) == 1 for y in range(GRID_H) for x in range(GRID_W)):
                return grid
            ok = collapse(grid)

# ======================================================
# Render
# ======================================================
def render(grid):
    img = Image.new("RGB", (GRID_W * CELL_SIZE, GRID_H * CELL_SIZE))

    for y in range(GRID_H):
        for x in range(GRID_W):
            color = COLORS[next(iter(grid[y][x]))]
            for dy in range(CELL_SIZE):
                for dx in range(CELL_SIZE):
                    img.putpixel(
                        (x * CELL_SIZE + dx, y * CELL_SIZE + dy),
                        color,
                    )

    img.show()
    img.save("grover_wfc_qi.png")

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    render(run_wfc())
