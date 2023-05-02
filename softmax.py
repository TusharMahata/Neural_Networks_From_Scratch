'''
import math
import numpy as np

layer_outputs = [-4.8, 1.21, -2.385]

E = math.e


exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

# using numpy

exp_values = np.exp(layer_outputs)
norm_values = exp_values / sum(exp_values)
print(norm_values)
print(sum(norm_values))


# For Batch inputs
import numpy as np
layer_outputs = [[1.2, 34, 4.3],
                [0.1, -3.2, -2],
                [1.41, 1.072, -1.773]]

exp_values = np.exp(layer_outputs)
print(exp_values)

# Sum per row and axis 0 for per column
# use keepdims to make dims column based

print(np.sum(layer_outputs, axis=1, keepdims=True))


norm_values = exp_values / np.sum(exp_values, axis = 0, keepdims=True)
print(norm_values)

'''
# Appling on neural network

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.baises = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        norm_value = exp_values / np.sum(exp_values, axis = 0, keepdims=True)
        self.output = norm_value  



X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:10])