import numpy as np
# from Classes.Neuron import *
class Layer():
    def __init__(self, begin_neuron_count, end_neuron_count, input_v=None):
        self.begin_neuron_count = begin_neuron_count
        self.end_neuron_count = end_neuron_count
        self.input_vector = np.array([1], input_v) #1 is because of mean term
        self.weight_matrix = np.transpose(np.random.rand(self.begin_neuron_count + 1, self.end_neuron_count))
        self.loss_derivative_matrix = np.zeros(np.shape(self.weight_matrix ))
        self.grad_vect = np.zeros((self.end_neuron_count,1))
        if self.input_vector != None:
           self.output_vector = np.matmul(self.input_vector, self.weight_matrix)

    def take_input(self, input_v):
        self.input_vector = np.array([1], input_v)

    def weight_matrix_update(self, l_rate):  # put the weights
        self.weight_matrix -= l_rate*self.loss_derivative_matrix

    def layer_output(self):
        self.output_vector = np.matmul(self.input_vector, self.weight_matrix)

    def loss_derivative(self, mat):
        self.loss_derivative_matrix = mat

    def grad_vector(self,vect):
        self.grad_vect = vect