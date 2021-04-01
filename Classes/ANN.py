from Classes.Layer import*
from Classes.Neuron import*

class ANN():
    def __init__(self, num_input, num_output, num_layer):
        self.num_input = num_input
        self.num_output = num_output
        self.num_layer = num_layer
        self.layers = []
        for layer in range(num_layer):
            self.neurons.append(Layer(self.node_count))

    def add_layer(self, num_new_layer):
        for layer in range(num_new_layer):
            self.layers = self.layers.append(Layer(self.inputs, self.weights))
        self.num_layer += num_new_layer

    def remove_layer(self,num_remove_layer):
        for layer in range(num_remove_layer):
            self.layers = self.layers.append(Layer(self.inputs, self.weights))
        self.num_layer -= num_remove_layer

    def forward_pass(self, flat_image):  # takes flat image calculates the output once, using the current weights
        self.layers[1].input_vector = flat_image
        for i in range(1, len(self.layers)):
            self.layer[i].input_vector = self.layer[i-1].output_vector

    def back_prop(self, result):

        pass
    def train(self, flat_image):  # takes one flatten image, updates weights using forward_pass and back_prop

        pass
