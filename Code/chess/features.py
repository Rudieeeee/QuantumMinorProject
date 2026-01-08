import chess
import numpy as np


# =============================
# Piece values
# =============================

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}


# =============================
# Basic material
# =============================

def material_balance(board: chess.Board) -> float:
    score = 0
    for piece, value in PIECE_VALUES.items():
        score += len(board.pieces(piece, chess.WHITE)) * value
        score -= len(board.pieces(piece, chess.BLACK)) * value
    return np.tanh(score / 10.0)


# =============================
# Positional features
# =============================

def center_control(board: chess.Board) -> float:
    center = [chess.D4, chess.E4, chess.D5, chess.E5]
    score = 0
    for sq in center:
        if board.is_attacked_by(chess.WHITE, sq):
            score += 1
        if board.is_attacked_by(chess.BLACK, sq):
            score -= 1
    return score / 4.0


def mobility(board: chess.Board) -> float:
    turn = board.turn

    board.turn = chess.WHITE
    white_moves = board.legal_moves.count()

    board.turn = chess.BLACK
    black_moves = board.legal_moves.count()

    board.turn = turn

    return np.tanh((white_moves - black_moves) / 20.0)


def bishop_pair(board: chess.Board) -> float:
    white = len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2
    black = len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2
    return float(white) - float(black)


def king_castled(board: chess.Board) -> float:
    def castled(color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0
        rank = chess.square_rank(king_sq)
        file = chess.square_file(king_sq)
        return int(rank in (0, 7) and file in (2, 6))

    return castled(chess.WHITE) - castled(chess.BLACK)


# =============================
# Pawn structure
# =============================

def passed_pawns(board: chess.Board) -> float:
    def count_passed(color):
        pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)

        passed = 0
        for p in pawns:
            file = chess.square_file(p)
            rank = chess.square_rank(p)

            for ep in enemy_pawns:
                ef = chess.square_file(ep)
                er = chess.square_rank(ep)

                if abs(ef - file) <= 1:
                    if (color == chess.WHITE and er > rank) or \
                       (color == chess.BLACK and er < rank):
                        break
            else:
                passed += 1

        return passed

    return np.tanh((count_passed(chess.WHITE) - count_passed(chess.BLACK)) / 2.0)


def doubled_pawns(board: chess.Board) -> float:
    def count(color):
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(p) for p in pawns]
        return len(files) - len(set(files))

    return np.tanh((count(chess.BLACK) - count(chess.WHITE)) / 4.0)


# =============================
# Side to move
# =============================

def side_to_move(board: chess.Board) -> float:
    return 1.0 if board.turn == chess.WHITE else -1.0


# =============================
# Final feature vector
# =============================

def extract_features(board: chess.Board) -> np.ndarray:
    """
    Feature vector (size = 9)
    """
    return np.array([
        material_balance(board),   # 0
        center_control(board),     # 1
        mobility(board),           # 2
        bishop_pair(board),        # 3
        king_castled(board),       # 4
        passed_pawns(board),       # 5
        doubled_pawns(board),      # 6
        side_to_move(board),       # 7
        1.0                         # 8 bias feature
    ], dtype=np.float32)
