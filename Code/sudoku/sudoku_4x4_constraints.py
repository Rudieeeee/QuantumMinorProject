from sudoku_4x4_generator import generate_puzzle

def index_unknowns(board):
    unknown_map = {}
    index = 0
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0:
                unknown_map[(r, c)] = index
                index += 1
    return unknown_map

def neighbors(r, c):
    neigh = set()

    # Row + column
    for i in range(4):
        neigh.add((r, i))
        neigh.add((i, c))

    # 2x2 block
    br = (r // 2) * 2
    bc = (c // 2) * 2
    for i in range(br, br + 2):
        for j in range(bc, bc + 2):
            neigh.add((i, j))

    neigh.remove((r, c))
    return neigh

def generate_value_constraints(board, unknown_map):
    value_constraints = set()

    for (r, c), u_idx in unknown_map.items():
        for nr, nc in neighbors(r, c):
            if board[nr][nc] != 0:
                prohibited = board[nr][nc] - 1
                value_constraints.add((u_idx, prohibited))

    return sorted(value_constraints)

def minimal_value_constraints(board, unknown_map):
    constraints = []

    for (r, c), idx in unknown_map.items():
        seen = set()
        for nr, nc in neighbors(r, c):
            if board[nr][nc] != 0:
                seen.add(board[nr][nc] - 1)

        for v in seen:
            constraints.append((idx, v))

    return constraints


def generate_relative_constraints(unknown_map):
    relative_constraints = []
    unknowns = list(unknown_map.items())

    for i in range(len(unknowns)):
        (r1, c1), idx1 = unknowns[i]
        for j in range(i + 1, len(unknowns)):
            (r2, c2), idx2 = unknowns[j]

            same_row = r1 == r2
            same_col = c1 == c2
            same_block = (r1 // 2 == r2 // 2) and (c1 // 2 == c2 // 2)

            if same_row or same_col or same_block:
                relative_constraints.append((idx1, idx2))

    return relative_constraints

def minimal_relative_constraints(board, unknown_map):
    constraints = set()

    def add_unit_constraints(cells):
        unknowns = [unknown_map[c] for c in cells if c in unknown_map]
        if len(unknowns) <= 1:
            return

        leader = unknowns[0]
        for other in unknowns[1:]:
            constraints.add((leader, other))

    # Rows
    for r in range(4):
        add_unit_constraints([(r, c) for c in range(4)])

    # Columns
    for c in range(4):
        add_unit_constraints([(r, c) for r in range(4)])

    # Blocks
    for br in range(0, 4, 2):
        for bc in range(0, 4, 2):
            add_unit_constraints([
                (r, c)
                for r in range(br, br + 2)
                for c in range(bc, bc + 2)
            ])

    return sorted(constraints)

def generate_constraints(test=False):
    if (test):
        puzzle = [[3, 0, 4, 0], [0, 1, 0, 2], [0, 4, 0, 3], [2, 0, 1, 0]]
    else:
        puzzle = generate_puzzle(8)

    unknown_map = index_unknowns(puzzle)

    # value_constraints = generate_value_constraints(puzzle, unknown_map)
    # relative_constraints = generate_relative_constraints(unknown_map)

    value_constraints = minimal_value_constraints(puzzle, unknown_map)
    relative_constraints = minimal_relative_constraints(puzzle, unknown_map)

    print("Puzzle:")
    for row in puzzle:
        print(row)

    print("\nValue constraints:")
    print(value_constraints)

    print("\nRelative constraints:")
    print(relative_constraints)

    return value_constraints, relative_constraints

generate_constraints(test=True)
