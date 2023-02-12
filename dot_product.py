inputs = [1, 2, 3, 2.7]

weights = [
    [0.2, .3, .7, 2.1],
    [0.2, .3, 2.7, .1],
    [3.2, 1.3, .7, 5]
]

bias = [2,3,1]
layer_output = [] 
#test = zip(weights, bias)
#print(test)
for neuron_weights, neuron_bias in zip(weights, bias):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)