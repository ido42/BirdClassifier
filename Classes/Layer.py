import numpy as np
# from Classes.Neuron import *
class Layer():
    def __init__(self, begin_neuron_count, end_neuron_count, input=None):
        self.begin_neuron_count = begin_neuron_count
        self.end_neuron_count = end_neuron_count
        self.input_vector = input
        self.weight_matrix = np.transpose(np.random.rand(self.begin_neuron_count + 1, self.end_neuron_count))

        if self.input_vector != None:
           self.output_vector = np.matmul(self.input_vector, self.weight_matrix)

    def take_input(self, input):
        self.input_vector = input

    def weight_matrix_update(self, loss_derivative, l_rate):  # put the weights
        self.weight_matrix -= l_rate*loss_derivative

    def layer_output(self):
        self.output_vector = np.matmul(self.input_vector, self.weight_matrix)


 #   def add_neuron(self, num_new_neuron, inputs):  # num_new_neuron contains the number of new neurons to be added to layer
  #      self.begin_neuron_count =

   # def remove_neuron(self, num_remove_neuron):
    #    for neuron in range(
     #           num_remove_neuron):  # num_remove_neuron contains the number of new neurons to be added to layer
      #      self.neurons = np.delete(self.neurons, -1)  # deletes the last element in list, kind  of arbitrary
       # self.neuron_count -= num_remove_neuron