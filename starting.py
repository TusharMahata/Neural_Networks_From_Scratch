import sys
import numpy as np
import matplotlib

print('Python:', sys.version)
print('Numpy:', np.__version__)
print('Matplotlib:', matplotlib.__version__)

# simple neural

inputs = [1.2, 5.1, 3.2]
weights = [3.1, 2.1, 8.7]
bias = 3


output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+bias
print(output)

# 3 neuron with 4 input
# it's seems a layer
inputs = [1, 2, 3, 2.7]

weights1 = [0.2, .3, .7, 2.1]
weights2 = [0.2, .3, 2.7, .1]
weights3 = [3.2, 1.3, .7, 5]
#weights4 = [0.2, .3, .7, 2.1]

bias1 = 2
bias2 = 3
bias3 = 1

output_of_3_neuron = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights[2]+inputs[3]*weights1[3]+bias1, inputs[1]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2, inputs[2]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3
]
print(output_of_3_neuron)
