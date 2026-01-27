import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
import random

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 740
GRID_TOP = 140
GRID_GAP = 5

#colors
COLOR_BG = (30, 30, 30)
COLOR_OFF = (50, 50, 50)
COLOR_ON = (255, 200, 0)
COLOR_PANEL = (40, 40, 40)
COLOR_BUTTON = (100, 100, 100)
COLOR_BUTTON_ACTIVE = (200, 180, 0)
COLOR_TEXT = (255, 255, 255)

#init
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Lights Out (Pygame)")
font = pygame.font.SysFont(None, 28)

#state
grid_size = 5
grid = [[0]*grid_size for _ in range(grid_size)]
mode = "toggle"
dragged_cells = set()
elapsed_time = 0.0
current_solution = []
congrats_start_time = None
unsolvable_start_time = None

def draw_button(rect, text, active=False):
    color = COLOR_BUTTON_ACTIVE if active else COLOR_BUTTON
    pygame.draw.rect(screen, color, rect)
    txt_surf = font.render(text, True, COLOR_TEXT)
    txt_rect = txt_surf.get_rect(center=rect.center)
    screen.blit(txt_surf, txt_rect)

def draw_top_panel():
    panel_rect = pygame.Rect(0,0,WINDOW_WIDTH, GRID_TOP)
    pygame.draw.rect(screen, COLOR_PANEL, panel_rect)

    toggle_rect = pygame.Rect(20, 20, 100, 40)
    play_rect = pygame.Rect(140, 20, 100, 40)
    draw_button(toggle_rect, "Toggle", mode=="toggle")
    draw_button(play_rect, "Play", mode=="play")

    size_label = font.render(f"Size: {grid_size}", True, COLOR_TEXT)
    screen.blit(size_label, (260, 30))
    plus_rect = pygame.Rect(340, 20, 40, 40)
    minus_rect = pygame.Rect(390, 20, 40, 40)
    draw_button(plus_rect, "+")
    draw_button(minus_rect, "-")

    solve_rect = pygame.Rect(450, 20, 100, 40)
    draw_button(solve_rect, "Solve")

    generate_rect = pygame.Rect(390, 70, 160, 40)
    draw_button(generate_rect, "Generate Board")

    benchmark_rect = pygame.Rect(20, 100, 130, 30)
    draw_button(benchmark_rect, "Benchmark")

    time_msg = font.render(f"Time: {elapsed_time:.3f}ms", True, COLOR_TEXT)
    screen.blit(time_msg, (20, 70))

    return {
        "toggle": toggle_rect,
        "play": play_rect,
        "plus": plus_rect,
        "minus": minus_rect,
        "solve": solve_rect,
        "generate": generate_rect,
        "benchmark": benchmark_rect
    }

def tile_rect(r, c):
    size = (WINDOW_WIDTH - GRID_GAP*(grid_size+1)) // grid_size
    x = GRID_GAP + c*(size + GRID_GAP)
    y = GRID_TOP + GRID_GAP + r*(size + GRID_GAP)
    return pygame.Rect(x, y, size, size)

def toggle_play(r, c):
    for dr, dc in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            grid[nr][nc] ^= 1

def handle_drag(pos):
    for r in range(grid_size):
        for c in range(grid_size):
            rect = tile_rect(r, c)
            if rect.collidepoint(pos) and (r,c) not in dragged_cells:
                grid[r][c] ^= 1
                dragged_cells.add((r,c))

def handle_click(pos):
    pressed_light = False
    for r in range(grid_size):
        for c in range(grid_size):
            rect = tile_rect(r, c)
            if rect.collidepoint(pos):
                toggle_play(r,c)
                pressed_light = True
                break
    return pressed_light

def draw_grid():
    for r in range(grid_size):
        for c in range(grid_size):
            rect = tile_rect(r,c)
            color = COLOR_ON if grid[r][c] else COLOR_OFF
            pygame.draw.rect(screen, color, rect)

def generate_solvable_board(grid_size):
    for _ in range(random.randint(grid_size, grid_size**2)):
        r = random.randint(0, grid_size-1)
        c = random.randint(0, grid_size-1)
        toggle_play(r,c)

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
    x = x.reshape(n, m)

    solution_coords = []
    for r in range(n):
        for c in range(m):
            if x[r, c] == 1:
                solution_coords.append((r, c))
    return solution_coords

def congrats_message():
    global congrats_start_time
    if congrats_start_time is not None:
        if time.time() - congrats_start_time < 3:
            msg_surf = font.render("Congratulations!", True, (255,255,0))
            screen.blit(msg_surf, (270 - msg_surf.get_width()//2, 70))
        else:
            congrats_start_time = None

def unsolvable_message():
    global unsolvable_start_time
    if unsolvable_start_time is not None:
        if time.time() - unsolvable_start_time < 3:
            msg_surf = font.render("Board is unsolvable!", True, (255,0,0))
            screen.blit(msg_surf, (270 - msg_surf.get_width()//2, 70))
        else:
            unsolvable_start_time = None

def benchmark():
    results = {}
    runs_per_size = 100

    for grid_size in range(3, 11):
        times = []

        for _ in range(runs_per_size):
            grid = [[0]*grid_size for _ in range(grid_size)]
            generate_solvable_board(grid_size)
            board_copy = [row[:] for row in grid]

            start = time.perf_counter()
            solve_lights_out(board_copy)
            end = time.perf_counter()

            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)
        results[grid_size] = avg_time
        
        print(f"Grid {grid_size}x{grid_size}: {avg_time:.3f} ms")

    plot_benchmark(results)

def plot_benchmark(results):
    sizes = np.array(list(results.keys()))
    times = np.array(list(results.values()))

    # constant = (times[4] - times[0] * (sizes[4] / sizes[0])**6) / (1 - (sizes[4] / sizes[0])**6)
    # times = times - constant

    #power law
    coeffs = np.polyfit(np.log(sizes), np.log(times), 1)
    exponent = coeffs[0]
    fitted_times = np.exp(coeffs[1]) * sizes**exponent

    #n^6 (scaled to match)
    theory_times = times[0] * (sizes / sizes[0])**6

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, "o-", label="Measured Average Time")
    plt.plot(sizes, fitted_times, "--", label=f"Fitted Curve ~ n^{exponent:.2f}")
    plt.plot(sizes, theory_times, ":", label=f"Theoretical ~ n^{6}")

    plt.xlabel("Grid Size (n x n)")
    plt.ylabel("Average Solve Time (ms)")
    plt.title("Lights Out Solver Benchmark")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

running = True
mouse_down = False
clock = pygame.time.Clock()

def draw_solution(current_solution):
    for r,c in current_solution:
        rect = tile_rect(r,c)
        pygame.draw.circle(screen, (255,0,0), rect.center, rect.width//6)

while running:
    screen.fill(COLOR_BG)
    button_rects = draw_top_panel()
    draw_grid()
    draw_solution(current_solution)
    congrats_message()
    unsolvable_message()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
            mouse_down = True
            dragged_cells.clear()
            pos = event.pos

            if button_rects["toggle"].collidepoint(pos):
                mode = "toggle"
            elif button_rects["play"].collidepoint(pos):
                mode = "play"
            elif button_rects["plus"].collidepoint(pos):
                grid_size = min(10, grid_size+1)
                grid = [[0]*grid_size for _ in range(grid_size)]
            elif button_rects["minus"].collidepoint(pos):
                grid_size = max(3, grid_size-1)
                grid = [[0]*grid_size for _ in range(grid_size)]
            elif button_rects["solve"].collidepoint(pos):
                start_time = time.time()
                solution = solve_lights_out(grid)
                elapsed_time = (time.time() - start_time) * 1000
                if solution is None:
                    unsolvable_start_time = time.time()
                    congrats_start_time = None
                    current_solution = []
                else:
                    current_solution = solution
                print(f"Solver ran in {elapsed_time:.3f}s")
            elif button_rects["generate"].collidepoint(pos):
                grid = [[0]*grid_size for _ in range(grid_size)]
                generate_solvable_board(grid_size)
                current_solution = []
            elif button_rects["benchmark"].collidepoint(pos):
                benchmark()
            else:
                if mode == "toggle":
                    handle_drag(pos)
                else:
                    light_pressed = handle_click(pos)
                    if light_pressed == True and all(grid[r][c]==0 for r in range(grid_size) for c in range(grid_size)):
                        congrats_start_time = time.time()
                        unsolvable_start_time = None
                        current_solution = []

        elif event.type == pygame.MOUSEBUTTONUP and event.button==1:
            mouse_down = False

        elif event.type == pygame.MOUSEMOTION and mouse_down:
            if mode == "toggle":
                handle_drag(event.pos)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
