from Classes.Neuron import *
class Layer():
    def __init__(self,node_count,layer_no):
        self.node_count = node_count
        self.layer_no = layer_no
        self. neurons=[]
        for neuron in range(node_count):
            self.neurons.append(Neuron(self.inputs, self.weights))




