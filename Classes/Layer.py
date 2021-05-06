import numpy as np

class Layer:
    def __init__(self, begin_neuron_count, end_neuron_count, activation='relu'): #, input_v=None):
        self.activation = activation
        self.begin_neuron_count = begin_neuron_count
        self.end_neuron_count = end_neuron_count
        self.bias = np.random.uniform(-1,1,(end_neuron_count, 1))
        self.weight_matrix = np.random.uniform(-1,1,(self.end_neuron_count, self.begin_neuron_count))
        self.loss_derivative_matrix = np.zeros(np.shape(self.weight_matrix))


    def forward(self, input_v):
        input_v = np.maximum(0, input_v)
        self.input_vector = input_v
        self.output_vector = np.nan_to_num(np.matmul(self.weight_matrix, self.input_vector) + self.bias)
        if self.activation == 'relu':
            self.output_vector = np.maximum(0, self.output_vector)
        else:
            self.output_vector = np.nan_to_num(np.exp(self.output_vector) / np.sum(np.exp(self.output_vector)))


    def backward(self, grad, l_rate, gradW=None):
        self.gradB = grad
        if gradW is None:
            self.gradW = np.matmul(self.input_vector, grad.transpose())
        else:
            self.gradW = gradW # gradW is only provided to the output layer
        self.grad = np.matmul(self.weight_matrix.transpose(), grad)
        # update parameters
        self.weight_matrix = self.weight_matrix - (l_rate * self.gradW).transpose()
        self.bias = self.bias - (l_rate * self.gradB)