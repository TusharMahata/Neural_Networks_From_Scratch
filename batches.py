import numpy as np


inputs  = [[1, 2, 3, 4.2],
           [2.4, 5.2, -3, 0.1],
           [-1.3, -3.2, 0, 4]]

weights = [[0.2, 0.2, 0.1, 2],
           [-2, -7.3, 1, 5],
           [9, 2, 5.5, 0.3]]

baises = [4, .9, .1]
# making Transpose for avioding shape error during dot multiplication
# biases applied here parallelly 
output = np.dot(inputs, np.array(weights).T) + baises
print(output)

# creating multilayer

weights2 = [[2,1,4],
            [-1,.3, -5],
            [.5, -.1, 3]]

baises2 = [1, .1, -1]
layer2_output = np.dot(output, np.array(weights2).T) + baises2
print(layer2_output)
