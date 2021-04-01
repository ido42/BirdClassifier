import numpy as np
from Classes.Neuron import *
class Layer():
    def __init__(self, neuron_count):
        self.neuron_count = neuron_count
        self. neurons = []
        for neuron in range(neuron_count):
            self.neurons.append(Neuron(self.inputs))
        self.input_vector = self.neurons[1].inputs  # all neurons in that layer take the same inputs

        self.MatrixUpdate()
        self.output_vector = np.matmul(self.input_vector, self.layer_weight_matrix)

    def MatrixUpdate(self): # put the weights
        num_neuron = len(self.neurons)  # neurons in this layer
        num_next_neuron = len(self.neurons.in_vector)  # neurons in the next layer
        l_mat = []
        for n in range(num_neuron):
            l_mat = np.append(l_mat, self.neurons.inputs)
        l_mat = np.reshape(l_mat, (num_neuron, num_next_neuron))
        self.layer_weight_matrix = l_mat

    def add_neuron(self, num_new_neuron):  # num_new_neuron contains the number of new neurons to be added to layer
        for neuron in range(num_new_neuron):
            self.neurons = self.neurons.append(Neuron(self.inputs))
        self.neuron_count += num_new_neuron

    def remove_neuron(self, num_remove_neuron):
        for neuron in range(
                num_remove_neuron):  # num_remove_neuron contains the number of new neurons to be added to layer
            self.neurons = np.delete(self.neurons, -1)  # deletes the last element in list, kind  of arbitrary
        self.node_count -= num_remove_neuron

    def layer_output(self):
        self.output_vector = np.matmul(self.input_vector, self.layer_weight_matrix)


