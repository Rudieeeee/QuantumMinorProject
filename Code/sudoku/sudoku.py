"""Sudoku/Latin-square solver with classical and quantum (Grover) backends.

This script provides a small, well-typed backtracking solver that
handles N x N puzzles where each row and each column must contain the
numbers 1..N exactly once. When N is a perfect square (for example
N=4 or N=9) the solver also enforces the standard Sudoku box
constraint (square subgrids of size sqrt(N)).

For 2x2 puzzles, a quantum Grover's algorithm solver is also available.

Includes example puzzles for 2x2, 3x3 (Latin-square style) and a
4x4 Sudoku example. Run as: python Code/sudoku.py
"""
import math
from typing import List, Optional, Tuple
from quantum_grover_2x2 import grover_solve_2x2


Grid = List[List[int]]


def print_grid(grid: Grid) -> None:
    for row in grid:
        print(" ".join(str(x) if x != 0 else "." for x in row))


def find_empty(grid: Grid) -> Optional[Tuple[int, int]]:
    n = len(grid)
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                return i, j
    return None


def is_valid(grid: Grid, r: int, c: int, val: int) -> bool:
    n = len(grid)
    # row
    if any(grid[r][j] == val for j in range(n)):
        return False
    # column
    if any(grid[i][c] == val for i in range(n)):
        return False
    # If n is a perfect square, also enforce subgrid (box) constraint
    root = int(math.isqrt(n))
    if root * root == n:
        br = (r // root) * root
        bc = (c // root) * root
        for i in range(br, br + root):
            for j in range(bc, bc + root):
                if grid[i][j] == val:
                    return False
    return True


def solve_backtracking(grid: Grid) -> Optional[Grid]:
    """Solve the grid in-place using backtracking; return a solved copy or None."""
    loc = find_empty(grid)
    if loc is None:
        return [row[:] for row in grid]

    r, c = loc
    n = len(grid)
    for v in range(1, n + 1):
        if is_valid(grid, r, c, v):
            grid[r][c] = v
            sol = solve_backtracking(grid)
            if sol is not None:
                return sol
            grid[r][c] = 0
    return None


def solve_and_print(grid: Grid) -> None:
    print("Puzzle:")
    print_grid(grid)
    sol = solve_backtracking([row[:] for row in grid])
    if sol is None:
        print("No solution found.")
    else:
        print("Solution:")
        print_grid(sol)


def solve_2x2_with_grover(puzzle: Grid) -> None:
    """Solve a 2x2 puzzle using Grover's algorithm (if Qiskit available)."""
    print("Puzzle:")
    print_grid(puzzle)
    print("\nAttempting quantum Grover solver...")
    sol = grover_solve_2x2([row[:] for row in puzzle])
    if sol is None:
        print("Grover solver failed or Qiskit unavailable; falling back to classical.")
        solve_and_print([row[:] for row in puzzle])


def example_2x2() -> Grid:
    # Simple 2x2 puzzle: top-left given as 1, rest empty
    # Unique solution: [[1,2],[2,1]]
    return [[1, 0], [0, 0]]


def example_3x3() -> Grid:
    # Example Latin-square-style 3x3 puzzle with three empties
    return [[0, 2, 3], [3, 0, 2], [2, 3, 0]]


def example_4x4() -> Grid:
    # 4x4 Sudoku example (uses 2x2 boxes). Zeros are empty cells.
    # This puzzle is solvable; solution should be a valid 4x4 Sudoku.
    return [
        [1, 0, 0, 4],
        [0, 4, 1, 0],
        [0, 0, 2, 1],
        [2, 0, 0, 3],
    ]


def main() -> None:
    print("== 2x2 example (Classical) ==")
    solve_and_print(example_2x2())
    print()
    print("== 2x2 example (Quantum Grover) ==")
    solve_2x2_with_grover(example_2x2())
    print()
    print("== 3x3 example ==")
    solve_and_print(example_3x3())
    print()
    print("== 4x4 example ==")
    solve_and_print(example_4x4())


if __name__ == "__main__":
    main()
