import numpy as np
#single neuron, contains inputs and weights,
# calculates output (size of the weight vector)=(size of the input vector)+1 because bias

class Neuron():
    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.in_vector = np.append([1], self.inputs)
        self.weights = weights
        self.output = np.matmul(self.inputs, self.weights)
        self.sigm_output = 1/(1+np.exp(self.output))
