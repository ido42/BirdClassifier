from Classes.Layer import*


class ANN():
    def __init__(self, num_layer, learning_rate, layers_neurons):  # layers_neurons is a list
        self.num_layer = num_layer
        self.l_rate = learning_rate
        self.layers = []
        for l in range(num_layer):
            self.layers.append(Layer(layers_neurons[l], layers_neurons[l+1]))

    def forward_pass(self, flat_image):  # takes flat image calculates the output once, using the current weights
        self.layers[0].take_input(flat_image)
        self.layers[0].layer_output()
        for i in range(1, len(self.layers)):
            out = self.layers[i-1].output_vector
            #sigm_out=1/(1+np.exp(out))  #sigmoid activation
            softmax_out = np.exp(out)/sum(np.exp(out))
            self.layers[i].take_input(softmax_out)
            self.layers[i].layer_output()

    def back_prop(self, result):  # result is in the form of a vector which is the same size with output
        #cross-entropy cost function
        loss=result-self.layers[-1].output_vector
        inp=np.matmul(np.transpose(self.layers[-1].input_vector),loss)
        self.layers[-1].weight_matrix_update(inp,self.l_rate)
        for layer in range(len(self.layers)):
            for neuron in range(self.layers[layer].begin_neuron_count):
            loss =
       # for i in range(len(self.layers)):
        #    for j in range(len(self.layers.output_vector)):
         #       sum_exp = sum(np.exp())
          #  dlast= (result/self.layers[-1].output_vector)*()

        return error

    def weight_decay(self,):
        pass



