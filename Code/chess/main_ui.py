import pygame
import torch
import chess

from chess_engine import ChessEngine
from search import select_best_move

# Feature extractors
from features import extract_features              # classical (9)
from features_qml import extract_qml_features      # quantum (2)

from ui_pygame import (
    draw_board,
    load_piece_images,
    square_from_mouse,
    WINDOW_SIZE
)

from qml_model import QMLChessModel
from classical_model import ClassicalChessModel


# =============================
# Bot selection menu
# =============================

def menu(screen):
    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 32)

    while True:
        screen.fill((30, 30, 30))

        title = font.render("QML CHESS", True, (255, 255, 255))
        q_text = small_font.render("Press Q  - Quantum Bot (2 qubits)", True, (200, 200, 200))
        c_text = small_font.render("Press C  - Classical Bot", True, (200, 200, 200))

        screen.blit(title, (WINDOW_SIZE // 2 - title.get_width() // 2, 120))
        screen.blit(q_text, (WINDOW_SIZE // 2 - q_text.get_width() // 2, 240))
        screen.blit(c_text, (WINDOW_SIZE // 2 - c_text.get_width() // 2, 280))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return "quantum"
                if event.key == pygame.K_c:
                    return "classical"


# =============================
# Load model + feature function
# =============================

def load_model_and_features(choice):
    """
    Returns:
        model      - evaluation model
        feature_fn - correct feature extractor
    """
    if choice == "quantum":
        model = QMLChessModel(num_qubits=2)
        model.model.load_state_dict(torch.load("qml_weights.pt"))
        model.model.eval()
        return model, extract_qml_features

    if choice == "classical":
        model = ClassicalChessModel(input_dim=9)
        model.load_state_dict(torch.load("classical_weights.pt"))
        model.eval()
        return model, extract_features

    raise ValueError("Invalid model choice")


# =============================
# Game loop
# =============================

def game_loop(screen, model, feature_fn):
    engine = ChessEngine()
    selected_square = None
    clock = pygame.time.Clock()

    while not engine.is_game_over():
        clock.tick(60)
        draw_board(engine.board, screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            # Human move (White)
            if event.type == pygame.MOUSEBUTTONDOWN and engine.turn():
                square = square_from_mouse(event.pos)

                if selected_square is None:
                    piece = engine.board.piece_at(square)
                    if piece and piece.color == chess.WHITE:
                        selected_square = square
                else:
                    legal_move = None
                    for move in engine.board.legal_moves:
                        if move.from_square == selected_square and move.to_square == square:
                            legal_move = move
                            break

                    if legal_move:
                        engine.push_move(legal_move.uci())

                    selected_square = None

        # Bot move (Black)
        if not engine.turn() and not engine.is_game_over():
            move = select_best_move(
                engine,
                model,
                feature_fn
            )
            engine.push_move(move)

    # Game over screen
    font = pygame.font.SysFont(None, 48)
    result_text = font.render(
        f"Game Over: {engine.game_result()}",
        True,
        (255, 255, 255)
    )
    screen.blit(
        result_text,
        (
            WINDOW_SIZE // 2 - result_text.get_width() // 2,
            WINDOW_SIZE // 2 - result_text.get_height() // 2
        )
    )
    pygame.display.flip()
    pygame.time.wait(4000)


# =============================
# Main entry point
# =============================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("QML Chess")

    load_piece_images()

    choice = menu(screen)
    model, feature_fn = load_model_and_features(choice)

    game_loop(screen, model, feature_fn)

    pygame.quit()


if __name__ == "__main__":
    main()
