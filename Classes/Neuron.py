import numpy as np
# single neuron, contains inputs, weights are initially random
# calculates output (size of the weight vector)=(size of the input vector)+1 because bias
#input row vector, weights column vector
class Neuron():
    def __init__(self, inputs):
        self.inputs = inputs
        self.in_vector = np.append([1], self.inputs)  # add bias term
        self.weights = np.transpose(np.random.rand(inputs.size+1,1))
        self.output = np.matmul(self.inputs, self.weights)
        self.sigm_output = 1/(1+np.exp(self.output))

    def update_weights(self, new_weights):
        self.weights = new_weights
