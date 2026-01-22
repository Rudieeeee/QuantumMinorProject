# import numpy as np

# def lights_out_solver(board):
#     """
#     Solve Lights Out using Gaussian elimination over GF(2).
#     board: 2D list or numpy array of 0s and 1s.
#     Returns a solution matrix of presses (0/1), or None if no solution.
#     """
#     board = np.array(board, dtype=int)
#     n, m = board.shape
#     size = n * m

#     # Helper to convert 2D index to 1D
#     def idx(r, c):
#         return r * m + c

#     # Build coefficient matrix A and vector b
#     A = np.zeros((size, size), dtype=int)
#     b = board.flatten()

#     for r in range(n):
#         for c in range(m):
#             i = idx(r, c)
#             # This button toggles itself and its neighbors
#             for dr, dc in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
#                 rr, cc = r + dr, c + dc
#                 if 0 <= rr < n and 0 <= cc < m:
#                     j = idx(rr, cc)
#                     A[j, i] = 1  # pressing i affects light j

#     # Gaussian elimination over GF(2)
#     A = np.concatenate([A, b.reshape(-1, 1)], axis=1)
#     rows, cols = A.shape
#     col = 0

#     for row in range(rows):
#         if col >= cols - 1:
#             break

#         # Find pivot
#         pivot = None
#         for r in range(row, rows):
#             if A[r, col] == 1:
#                 pivot = r
#                 break
#         if pivot is None:
#             col += 1
#             continue

#         # Swap rows
#         A[[row, pivot]] = A[[pivot, row]]

#         # Eliminate other rows
#         for r in range(rows):
#             if r != row and A[r, col] == 1:
#                 A[r] ^= A[row]  # XOR row

#         col += 1

#     # Check for inconsistency
#     for r in range(rows):
#         if np.all(A[r, :-1] == 0) and A[r, -1] == 1:
#             return None  # No solution

#     # Extract solution (free variables set to 0)
#     x = np.zeros(size, dtype=int)
#     for r in range(rows):
#         pivot_col = np.where(A[r, :-1] == 1)[0]
#         if len(pivot_col) > 0:
#             x[pivot_col[0]] = A[r, -1]

#     return x.reshape(n, m)

# # Example usage:
# board = [
#     [1, 1, 1],
#     [1, 1, 0],
#     [1, 0, 1]
# ]

# solution = lights_out_solver(board)
# print(solution)

'''
side="top" → default, put it at the top

side="bottom" → put at bottom

side="left" → left side

side="right" → right side

padx=10 → horizontal padding

pady=10 → vertical padding

fill="x" → stretch horizontally

fill="y" → stretch vertically

expand=True → let it expand to fill extra space
'''

import tkinter as tk
from tkinter import messagebox
import numpy as np
import time


class LightsOutApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lights Out Solver (Gaussian Elimination)")

        self.size = tk.IntVar(value=5)

        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)

        tk.Label(top_frame, text="Grid size:").pack(side=tk.LEFT)
        tk.Spinbox(top_frame, from_=2, to=10, width=5, textvariable=self.size).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Create Grid", command=self.create_grid).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Solve", command=self.solve).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.info_label = tk.Label(root, text="")
        self.info_label.pack(pady=5)

        self.grid = []
        self.rectangles = []
        self.solution = None
        self.cell_size = 60

        self.create_grid()

    def create_grid(self):
        self.canvas.delete("all")
        self.grid = []
        self.rectangles = []
        self.solution = None
        self.info_label.config(text="")

        n = self.size.get()
        canvas_size = min(500, max(300, n * self.cell_size))
        self.canvas.config(width=canvas_size, height=canvas_size)
        self.cell_size = canvas_size // n

        for r in range(n):
            row = []
            rect_row = []
            for c in range(n):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray")
                self.canvas.tag_bind(rect, "<Button-1>", lambda e, r=r, c=c: self.toggle_cell(r, c))
                row.append(0)
                rect_row.append(rect)
            self.grid.append(row)
            self.rectangles.append(rect_row)

    def toggle_cell(self, r, c):
        self.grid[r][c] ^= 1
        color = "yellow" if self.grid[r][c] == 1 else "black"
        self.canvas.itemconfig(self.rectangles[r][c], fill=color)

    def draw_solution(self):
        self.canvas.delete("solution")
        if self.solution is None:
            return

        n = self.size.get()
        for r in range(n):
            for c in range(n):
                if self.solution[r][c] == 1:
                    cx = c * self.cell_size + self.cell_size // 2
                    cy = r * self.cell_size + self.cell_size // 2
                    radius = self.cell_size // 6
                    self.canvas.create_oval(
                        cx - radius, cy - radius,
                        cx + radius, cy + radius,
                        fill="red", tags="solution"
                    )

    def solve(self):
        start_time = time.perf_counter()
        solution = solve_lights_out(self.grid)
        end_time = time.perf_counter()

        if solution is None:
            messagebox.showerror("No solution", "This configuration has no solution.")
            return

        self.solution = solution
        self.draw_solution()

        elapsed_ms = (end_time - start_time) * 1000
        self.info_label.config(text=f"Solved in {elapsed_ms:.3f} ms")


def solve_lights_out(board):
    board = np.array(board, dtype=int)
    n, m = board.shape
    size = n * m

    def idx(r, c):
        return r * m + c

    A = np.zeros((size, size), dtype=int)
    b = board.flatten()

    for r in range(n):
        for c in range(m):
            i = idx(r, c)
            for dr, dc in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < m:
                    j = idx(rr, cc)
                    A[j, i] = 1

    M = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    rows, cols = M.shape
    col = 0

    for row in range(rows):
        if col >= cols - 1:
            break

        pivot = None
        for r in range(row, rows):
            if M[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            col += 1
            continue

        M[[row, pivot]] = M[[pivot, row]]

        for r in range(rows):
            if r != row and M[r, col] == 1:
                M[r] ^= M[row]

        col += 1

    for r in range(rows):
        if np.all(M[r, :-1] == 0) and M[r, -1] == 1:
            return None

    x = np.zeros(size, dtype=int)
    for r in range(rows):
        pivot_cols = np.where(M[r, :-1] == 1)[0]
        if len(pivot_cols) > 0:
            x[pivot_cols[0]] = M[r, -1]

    return x.reshape(n, m)


if __name__ == "__main__":
    root = tk.Tk()
    app = LightsOutApp(root)
    root.mainloop()
