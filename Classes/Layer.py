from Classes.Neuron import *
class Layer():
    def __init__(self,neuron_count,layer_no):
        self.neuron_count = neuron_count
        self.layer_no = layer_no
        self. neurons = []
        for neuron in range(neuron_count):
            self.neurons.append(Neuron(self.inputs))

    def add_neuron(self, num_new_neuron):  # num_new_neuron contains the number of new neurons to be added to layer
        for neuron in range(num_new_neuron):
            self.neurons = self.neurons.append(Neuron(self.inputs))
        self.neuron_count += num_new_neuron

    def remove_neuron(self, num_remove_neuron):
        for neuron in range(num_remove_neuron):  # num_remove_neuron contains the number of new neurons to be added to layer
            self.neurons = np.delete(self.neurons, -1)  #deletes the last element in list, kindo of arbitrary
        self.node_count -= num_remove_neuron
