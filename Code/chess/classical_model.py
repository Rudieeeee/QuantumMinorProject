import torch
import torch.nn as nn


class ClassicalChessModel(nn.Module):
    def __init__(self, input_dim: int = 9):
        """
        Classical neural network for chess evaluation.

        input_dim must match the number of features.
        Output range: [-1, 1] (White advantage positive).
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass used during training.
        """
        return self.net(x)

    def predict(self, features):
        """
        Predict a scalar evaluation in [-1, 1].
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score = self.forward(x)

        return float(score.item())
