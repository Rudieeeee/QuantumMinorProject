import chess


class ChessEngine:
    def __init__(self):
        """
        Initialize a standard chess board.
        """
        self.board = chess.Board()

    # =============================
    # Game state control
    # =============================

    def reset(self):
        """
        Reset the game to the initial position.
        """
        self.board.reset()

    def copy(self):
        """
        Return a deep copy of the current engine state.
        """
        new_engine = ChessEngine()
        new_engine.board = self.board.copy()
        return new_engine

    # =============================
    # Board information
    # =============================

    def get_fen(self):
        """
        Return the current board position in FEN format.
        """
        return self.board.fen()

    def turn(self):
        """
        Return whose turn it is.
        True = White, False = Black
        """
        return self.board.turn

    def is_game_over(self):
        """
        Check if the game is over.
        """
        return self.board.is_game_over()

    def game_result(self):
        """
        Return the game result as a string.
        """
        if not self.board.is_game_over():
            return None
        return self.board.result()

    # =============================
    # Move handling
    # =============================

    def get_legal_moves(self):
        """
        Return a list of legal moves in UCI format.
        """
        return [move.uci() for move in self.board.legal_moves]

    def push_move(self, move_uci: str):
        """
        Apply a move to the board (UCI format).
        Automatically handles pawn promotion (defaults to queen).
        Raises ValueError if move is illegal.
        """
        move = chess.Move.from_uci(move_uci)

        # Auto-handle pawn promotion if missing
        piece = self.board.piece_at(move.from_square)
        if (
            piece
            and piece.piece_type == chess.PAWN
            and chess.square_rank(move.to_square) in [0, 7]
            and move.promotion is None
        ):
            move = chess.Move(
                move.from_square,
                move.to_square,
                promotion=chess.QUEEN
            )

        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")

        self.board.push(move)

    def pop_move(self):
        """
        Undo the last move.
        """
        self.board.pop()

    # =============================
    # Debug / display
    # =============================

    def display(self):
        """
        Print the board to the console.
        """
        print(self.board)
