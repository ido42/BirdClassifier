import numpy as np

class Neuron():

    def __init__(self, inputs, weights):
        self.inputs = inputs
        self.weights = weights
        self.output = np.matmul(self.inputs, self.weights)
        self.sigm_output = 1/(1+np.exp(self.output))
