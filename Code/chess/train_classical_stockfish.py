import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import chess

from features import extract_features
from classical_model import ClassicalChessModel


# =============================
# Load Stockfish dataset
# =============================

def load_stockfish_csv(filename, max_samples=500000):
    """
    Load Stockfish-evaluated positions from CSV.

    Expected columns:
    - FEN
    - Evaluation (centipawns or mate scores)
    """
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, filename)

    print("Loading CSV from:", csv_path)

    df = pd.read_csv(csv_path)
    print("Detected columns:", df.columns.tolist())

    X, y = [], []

    for _, row in df.iterrows():
        fen = row["FEN"]
        raw_eval = str(row["Evaluation"]).strip()

        # Skip mate evaluations like '#+6'
        if raw_eval.startswith("#"):
            continue

        board = chess.Board(fen)
        features = extract_features(board)

        # Convert centipawns → normalized score
        centipawns = float(raw_eval)
        pawns = centipawns / 100.0
        eval_score = np.tanh(pawns / 5.0)

        X.append(features)
        y.append([eval_score])

        if len(X) >= max_samples:
            break

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    print(f"Loaded {len(X)} usable positions")

    return X, y


# =============================
# Train classical model
# =============================

def train_classical():
    print("Loading Stockfish dataset...")
    X, y = load_stockfish_csv("chessData.csv")

    print(f"Dataset size: {len(X)} positions")

    model = ClassicalChessModel(input_dim=9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    epochs = 30

    print("Starting classical training...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(X)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Loss: {loss.item():.6f}"
        )

    torch.save(model.state_dict(), "classical_weights.pt")
    print("Training complete.")
    print("Saved weights to classical_weights.pt")


if __name__ == "__main__":
    train_classical()
