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

