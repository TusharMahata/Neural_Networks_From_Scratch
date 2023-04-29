import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [3, -2, -1, .01],
     [0.0, 0.3, -9, -1]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.baises = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises

layer1 = Layer_Dense(4, 5)
layer1.forward(X)

print(layer1.output)

layer2 = Layer_Dense(5, 2)
layer2.forward(layer1.output)
print(layer2.output)