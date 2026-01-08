import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer.primitives import Sampler

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector


class QMLChessModel:
    def __init__(self, num_qubits: int = 2):
        """
        Variational Quantum Neural Network for chess evaluation.

        num_qubits must match number of features.
        Output is a scalar in [-1, 1].
        """
        self.num_qubits = num_qubits

        self.sampler = Sampler()
        self.qnn = self._build_qnn()
        self.model = TorchConnector(self.qnn)

    def _build_qnn(self):
        # =============================
        # Parameters
        # =============================

        input_params = [Parameter(f"x{i}") for i in range(self.num_qubits)]
        weight_params = [
            Parameter(f"θ{i}") for i in range(self.num_qubits * 3)
        ]

        qc = QuantumCircuit(self.num_qubits)

        # =============================
        # Feature encoding
        # =============================

        for i in range(self.num_qubits):
            qc.ry(input_params[i] * np.pi, i)

        # =============================
        # Variational layer 1
        # =============================

        idx = 0
        for i in range(self.num_qubits):
            qc.ry(weight_params[idx], i)
            idx += 1

        # =============================
        # Ring entanglement (stronger than chain)
        # =============================

        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.num_qubits - 1, 0)

        # =============================
        # Variational layer 2
        # =============================

        for i in range(self.num_qubits):
            qc.ry(weight_params[idx], i)
            idx += 1

        # =============================
        # Second entanglement
        # =============================

        for i in range(0, self.num_qubits - 1, 2):
            qc.cx(i, i + 1)

        # =============================
        # Variational layer 3
        # =============================

        for i in range(self.num_qubits):
            qc.ry(weight_params[idx], i)
            idx += 1

        qc.measure_all()

        # =============================
        # QNN
        # =============================

        return SamplerQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            interpret=lambda x: x % 2,
            output_shape=2
        )

    def predict(self, features: np.ndarray) -> float:
        """
        Predict a scalar evaluation in [-1, 1].
        Positive = White advantage.
        """
        if features.shape[0] != self.num_qubits:
            raise ValueError(
                f"Expected {self.num_qubits} features, got {features.shape[0]}"
            )

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs = self.model(x)

        score = probs[0, 1] - probs[0, 0]
        return float(score.item())
