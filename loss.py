import math

print(math.log(3))

softmax_output = [.7, .1, .2]
target_output = [1, 0, 0]

# -log(y)*k
loss = -(math.log(softmax_output[0]) * target_output[0] + math.log(softmax_output[1]) * target_output[1] + math.log(softmax_output[2]) * target_output[2])

print(loss)