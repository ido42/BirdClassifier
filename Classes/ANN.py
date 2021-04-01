from Classes.Layer import*


class ANN():
    def __init__(self, num_layer, learning_rate, layers_neurons): #layers_neurons is a list
        self.num_layer = num_layer
        self.l_rate = learning_rate
        self.layers = []
        for l in range(num_layer):
            self.layers.append(Layer(layers_neurons[l], layers_neurons[l+1]))

    def forward_pass(self, flat_image):  # takes flat image calculates the output once, using the current weights
        self.layers[1].take_input(flat_image)

        for i in range(1, len(self.layers)):
            softmax_out = np.exp(self.layers[i-1].layer_output())/sum(np.exp(self.layers[i-1].layer_output()))
            self.layers[i].take_input(softmax_out)
        self.layers[-1].take_input(1/(np.exp(self.layers[-2].layer_output())/sum(np.exp(self.layers[-2].layer_output()))))

    def back_prop(self, result):  # result is in the form of a vector which is the same size with output
        error =
        return error


