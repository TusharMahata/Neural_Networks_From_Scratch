import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.baises = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)

layer1.forward(X)
print(layer1.output)

activation1 = Activation_ReLU()
activation1.forward(layer1.output)
print(activation1.output)

