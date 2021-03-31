from Classes.Layer import*
from Classes.Neuron import*

class ANN():
    def __init__(self, num_input, num_output, num_layer):
        self.num_input = num_input
        self.num_output = num_output
        self.num_layer = num_layer
        self.layers = []
        for layer in range(num_layer):
            self.neurons.append(Layer(self.node_count,self.layer_no))

    def add_layer(self, num_new_layer):
        for layer in range(num_new_layer):
            self.layers = self.layers.append(Layer(self.inputs, self.weights))
        self.num_layer += num_new_layer

    def remove_layer(self,num_remove_layer):
        for layer in range(num_remove_layer):
            self.layers = self.layers.append(Layer(self.inputs, self.weights))
        self.num_layer -= num_remove_layer

    def forward_pass(self):
        pass
    def back_prop(self):
        pass
    def train(self, flat_image):  # takes one flatten image, updates weights using forward_pass and back_prop
        pass
