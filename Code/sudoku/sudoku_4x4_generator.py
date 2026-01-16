import random

def find_empty(board):
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0:
                return r, c
    return None

def is_valid(board, r, c, num):
    if num in board[r]:
        return False
    if num in [board[i][c] for i in range(4)]:
        return False

    br = (r // 2) * 2
    bc = (c // 2) * 2
    for i in range(br, br + 2):
        for j in range(bc, bc + 2):
            if board[i][j] == num:
                return False

    return True

def count_solutions(board, limit=2):
    empty = find_empty(board)
    if not empty:
        return 1

    r, c = empty
    count = 0

    for num in range(1, 5):
        if is_valid(board, r, c, num):
            board[r][c] = num
            count += count_solutions(board, limit)
            board[r][c] = 0
            if count >= limit:
                break

    return count

def count_filled(board):
    return sum(1 for r in range(4) for c in range(4) if board[r][c] != 0)


def generate_full_board():
    board = [[0]*4 for _ in range(4)]

    def fill():
        empty = find_empty(board)
        if not empty:
            return True

        r, c = empty
        nums = list(range(1, 5))
        random.shuffle(nums)

        for num in nums:
            if is_valid(board, r, c, num):
                board[r][c] = num
                if fill():
                    return True
                board[r][c] = 0

        return False

    fill()
    return board

def generate_puzzle(unknowns):
    while True: 
        board = generate_full_board()
        cells = [(r, c) for r in range(4) for c in range(4)]
        random.shuffle(cells)

        for r, c in cells:
            if count_filled(board) <= unknowns:
                break

            backup = board[r][c]
            board[r][c] = 0

            if count_solutions(board) != 1:
                board[r][c] = backup

        if count_filled(board) == unknowns and count_solutions(board) == 1:
            return board
