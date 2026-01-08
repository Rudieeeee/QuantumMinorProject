import pygame
import chess
import os

SQUARE_SIZE = 75
WINDOW_SIZE = 600

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)

# Map pieces to filenames
PIECE_FILES = {
    chess.PAWN:   "p",
    chess.ROOK:   "r",
    chess.KNIGHT: "n",
    chess.BISHOP: "b",
    chess.QUEEN:  "q",
    chess.KING:   "k",
}

PIECE_IMAGES = {}


def load_piece_images():
    """
    Load and scale all piece images once.
    """
    base_path = os.path.join(os.path.dirname(__file__), "pieces")

    for color in ["w", "b"]:
        for piece, name in PIECE_FILES.items():
            filename = f"{color}{name}.png"
            path = os.path.join(base_path, filename)

            image = pygame.image.load(path).convert_alpha()
            image = pygame.transform.smoothscale(
                image, (SQUARE_SIZE, SQUARE_SIZE)
            )

            PIECE_IMAGES[(color, piece)] = image


def draw_board(board: chess.Board, screen):
    # Draw squares
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            color = LIGHT if (rank + file) % 2 == 0 else DARK

            rect = pygame.Rect(
                file * SQUARE_SIZE,
                rank * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE
            )
            pygame.draw.rect(screen, color, rect)

            piece = board.piece_at(square)
            if piece:
                color_key = "w" if piece.color == chess.WHITE else "b"
                image = PIECE_IMAGES[(color_key, piece.piece_type)]
                screen.blit(image, rect)

    pygame.display.flip()

def square_from_mouse(pos):
    """
    Convert mouse (x, y) position to a chess square.
    """
    x, y = pos
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    return chess.square(file, rank)
