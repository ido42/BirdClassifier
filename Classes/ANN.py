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
            relu_out = max(0,out)
            #softmax_out = np.exp(out)/sum(np.exp(out))
            self.layers[i].take_input(relu_out)
            self.layers[i].layer_output()

    def back_prop(self, result):  # result is in the form of a vector which is the same size with output
        #cross-entropy cost function
        softmax_out = np.exp(self.layers[-1].output_vector) / sum(np.exp(self.layers[-1].output_vector))
        grad_last = (softmax_out-result)
        der_loss = np.matmul(np.transpose(grad_last), self.layers[-2].output_vector)
        self.layers[-1].grad_vector(np.transpose(grad_last))
        self.layers[-1].loss_derivative(der_loss)
        self.layers[-1].weight_matrix_update(self.l_rate)

        for l in range(0, len(self.layers)-1, -1):
            grad = np.transpose(np.heaviside(self.layers[l].input_vector, 1)) * np.matmul(self.layers[l].weight_matrix, self.layers[l+1].grad_vect)
            der_loss = np.matmul(np.transpose(grad), self.layers[l-1].output_vector)
            self.layers[l].grad_vector(grad)
            self.layers[l].loss_derivative(der_loss)
            self.layers[l].weight_matrix_update(self.l_rate)
    def weight_decay(self):
        pass




