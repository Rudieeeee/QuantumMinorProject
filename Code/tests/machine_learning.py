import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

algorithm_globals.random_seed = 42
np.random.seed(42)

data = load_breast_cancer()
X = data.data
y = data.target

X = X[:, :2]   # mean radius & mean texture

scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

inputs = ParameterVector("x", 2)
weights = ParameterVector("θ", 6)

qc = QuantumCircuit(2)

# Feature map
qc.ry(inputs[0], 0)
qc.ry(inputs[1], 1)
qc.cx(0, 1)

# Variational layer
qc.ry(weights[0], 0)
qc.ry(weights[1], 1)
qc.cx(0, 1)

qc.ry(inputs[0], 0)
qc.ry(inputs[1], 1)

# Final variational layer
qc.ry(weights[2], 0)
qc.ry(weights[3], 1)
qc.ry(weights[4], 0)
qc.ry(weights[5], 1)

def interpret(x: int) -> int:
    return 0 if x < 2 else 1

sampler = StatevectorSampler()

qnn = SamplerQNN(
    circuit=qc,
    input_params=inputs,
    weight_params=weights,
    sampler=sampler,
    interpret=interpret,
    output_shape=2
)

optimizer = COBYLA(maxiter=300)

classifier = NeuralNetworkClassifier(
    neural_network=qnn,
    optimizer=optimizer,
    initial_point=np.random.rand(qnn.num_weights)
)

classifier.fit(X_train, y_train)

y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

train_acc = np.mean(y_train_pred == y_train)
test_acc = np.mean(y_test_pred == y_test)

print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)
