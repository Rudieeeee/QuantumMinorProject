import random
import numpy as np
from PIL import Image

TILES = ["grass", "sand", "water"]
ADJACENCY = {
    "grass": {"grass", "sand"},
    "sand": {"grass", "sand", "water"},
    "water": {"sand", "water"},
}

GRID_W, GRID_H = 90, 90


def new_grid():
    return [[set(TILES) for _ in range(GRID_W)] for _ in range(GRID_H)]


def entropy(cell):
    return len(cell)


def neighbors(x, y):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny


def propagate(grid, start):
    stack = [start]

    while stack:
        x, y = stack.pop()
        current = grid[y][x]

        for nx, ny in neighbors(x, y):
            allowed = set()
            for t in current:
                allowed |= ADJACENCY[t]

            before = grid[ny][nx]
            after = before & allowed

            if not after:
                return False

            if after != before:
                grid[ny][nx] = after
                stack.append((nx, ny))

    return True


def collapse(grid):
    # find lowest-entropy cell (>1)
    cells = [
        (x, y)
        for y in range(GRID_H)
        for x in range(GRID_W)
        if len(grid[y][x]) > 1
    ]

    if not cells:
        return True  # done

    x, y = min(cells, key=lambda c: entropy(grid[c[1]][c[0]]))
    choice = random.choice(list(grid[y][x]))
    grid[y][x] = {choice}

    return propagate(grid, (x, y))


def run_wfc():
    while True:
        grid = new_grid()
        ok = True

        while ok:
            if all(len(grid[y][x]) == 1 for y in range(GRID_H) for x in range(GRID_W)):
                return grid
            ok = collapse(grid)

        # restart on failure


def render(grid):
    size = 16
    img = Image.new("RGB", (GRID_W * size, GRID_H * size))

    colors = {
        "grass": (34, 139, 34),
        "sand": (194, 178, 128),
        "water": (30, 144, 255),
    }

    for y in range(GRID_H):
        for x in range(GRID_W):
            tile = next(iter(grid[y][x]))
            for dy in range(size):
                for dx in range(size):
                    img.putpixel((x * size + dx, y * size + dy), colors[tile])

    img.show()


if __name__ == "__main__":
    grid = run_wfc()
    render(grid)
