"""Classical Sudoku solver for 2x2 and 3x3 (Latin-square style).

This script provides a small, well-typed backtracking solver that
handles N x N puzzles where each row and each column must contain the
numbers 1..N exactly once. It supports N=2 and N=3 and includes example
puzzles and a simple CLI entry point (run with no args to solve examples).

Run as: python Code/sudoku_classical.py
"""
from typing import List, Optional, Tuple


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


def example_2x2() -> Grid:
    # Simple 2x2 puzzle: top-left given as 1, rest empty
    # Unique solution: [[1,2],[2,1]]
    return [[1, 0], [0, 0]]


def example_3x3() -> Grid:
    # Example Latin-square-style 3x3 puzzle with three empties
    return [[0, 2, 3], [3, 0, 2], [2, 3, 0]]


def main() -> None:
    print("== 2x2 example ==")
    solve_and_print(example_2x2())
    print()
    print("== 3x3 example ==")
    solve_and_print(example_3x3())


if __name__ == "__main__":
    main()
