import chess
import numpy as np

def extract_qml_features(board: chess.Board) -> np.ndarray:
    material = 0
    for piece, value in {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }.items():
        material += len(board.pieces(piece, chess.WHITE)) * value
        material -= len(board.pieces(piece, chess.BLACK)) * value

    mobility = board.legal_moves.count()

    return np.array([
        np.tanh(material / 10.0),
        np.tanh(mobility / 20.0)
    ], dtype=np.float32)
