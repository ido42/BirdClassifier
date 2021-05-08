import numpy as np
from numba import jit, cuda
import cupy as cp

class Layer:

    def __init__(self, begin_neuron_count, end_neuron_count, activation='relu'): #, input_v=None):
        self.activation = activation
        self.begin_neuron_count = begin_neuron_count
        self.end_neuron_count = end_neuron_count
        self.bias = np.random.uniform(-1,1,(1, end_neuron_count))
        self.weight_matrix = np.random.uniform(-1,1,(self.begin_neuron_count, self.end_neuron_count))


    def forward(self, input_v):
        input_v = np.maximum(0, input_v)
        self.input_vector = self.preventOF(input_v)
        self.output_vector = self.preventOF(np.matmul(self.input_vector, self.preventOF(self.weight_matrix)) + self.bias)
        if self.activation == 'relu':
            self.output_vector = np.maximum(0, self.output_vector)
        else:
            self.output_vector = np.true_divide(self.preventOF(np.exp(self.boundOut(self.output_vector))),
                                                np.sum(self.preventOF(np.exp(self.boundOut(self.output_vector)))))


    def backward(self, grad, l_rate):
        self.gradB = self.preventOF(grad)
        # if gradW is None:
        self.gradW = np.matmul(self.preventOF(grad).transpose(), self.input_vector)
        # else:
        #     self.gradW = gradW # gradW is only provided to the output layer
        self.grad = self.preventOF(np.matmul(self.preventOF(grad), self.weight_matrix.transpose()))
        # update parameters
        self.weight_matrix = self.weight_matrix - (l_rate * self.gradW.transpose())
        self.bias = self.bias - (l_rate * self.gradB)

    @staticmethod
    def preventOF(mat):  # to prevent overflows
        temp = np.where(mat < 1e-200, 0, mat)
        temp = np.where(0 < mat, temp, mat)
        return np.where(1e200 < temp, 1e200, temp)

    @staticmethod
    def boundOut(mat):
        temp = np.where(mat < -200, np.NINF, mat)
        return np.where(temp > 200, np.inf, temp)