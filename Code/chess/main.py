import torch
import chess

from chess_engine import ChessEngine
from search import select_best_move
from features import extract_features
from qml_model import QMLChessModel
from classical_model import ClassicalChessModel


# =============================
# Menu
# =============================

def menu():
    print("===================================")
    print("            QML CHESS               ")
    print("===================================")
    print("Choose bot:")
    print("1 - Classical (Stockfish-trained)")
    print("2 - Quantum (QML)")
    print()

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return "classical"
        if choice == "2":
            return "quantum"
        print("Invalid choice.")


# =============================
# Load model
# =============================

def load_model(choice):
    if choice == "quantum":
        model = QMLChessModel(num_qubits=2)
        model.model.load_state_dict(torch.load("qml_weights.pt"))
        model.model.eval()
        return model

    if choice == "classical":
        model = ClassicalChessModel(input_dim=9)
        model.load_state_dict(torch.load("classical_weights.pt"))
        model.eval()
        return model

    raise ValueError("Invalid model choice")


# =============================
# Move selection
# =============================

def select_best_move(engine, model):
    """
    Choose best move using evaluation model.
    White maximizes, Black minimizes.
    """
    best_move = None
    best_score = -1e9 if engine.turn() else 1e9

    for move in engine.get_legal_moves():
        engine.push_move(move)

        features = extract_features(engine.board)
        score = model.predict(features)

        engine.pop_move()

        if engine.turn():  # White
            if score > best_score:
                best_score = score
                best_move = move
        else:  # Black
            if score < best_score:
                best_score = score
                best_move = move

    return best_move


# =============================
# Main game loop
# =============================

def main():
    engine = ChessEngine()

    choice = menu()
    model = load_model(choice)

    print()
    print("You are WHITE")
    print("Enter moves in UCI format (e2e4)")
    print(engine.board)
    print()

    while not engine.is_game_over():
        if engine.turn():  # Human
            move = input("Your move: ").strip()
            try:
                engine.push_move(move)
            except ValueError as e:
                print(e)
                continue
        else:  # Bot
            print("Bot is thinking...")
            move = select_best_move(engine, model, extract_features)
            print(f"Bot plays: {move}")
            engine.push_move(move)

        print(engine.board)
        print()

    print("===================================")
    print("Game over!")
    print("Result:", engine.game_result())
    print("===================================")


if __name__ == "__main__":
    main()
