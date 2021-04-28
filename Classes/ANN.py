from Classes.Layer import*
from scipy.signal import convolve2d

class ANN():
    def __init__(self, num_layer, learning_rate, layers_neurons):  # layers_neurons is a list, len(layer_neurons)=num_layer+1,
        # first element of layer neurons should be the same with the length of flatten image
        self.num_layer = num_layer
        self.l_rate = learning_rate
        self.layers = []
        for l in range(num_layer-1):
            self.layers.append(Layer(layers_neurons[l], layers_neurons[l+1]))
        self.softmax_out = None

    def forward_pass(self, flat_image):  # takes flat image calculates the output once, using the current weights
        self.layers[0].take_input(flat_image)
        self.layers[0].layer_output()
        for i in range(1, len(self.layers)):
            out = self.layers[i-1].output_vector
            #sigm_out=1/(1+np.exp(out))  #sigmoid activation
            relu_out = np.maximum(0,out)
            #softmax_out = np.exp(out)/sum(np.exp(out))
            self.layers[i].take_input(relu_out)
            self.layers[i].layer_output()
        softmax_out = np.exp(self.layers[-1].output_vector) / sum(np.exp(self.layers[-1].output_vector))
        self.softmax_out=softmax_out
    def back_prop(self, result):  # result is in the form of a vector which is the same size with output
        #cross-entropy cost function
        cross_entropy = -np.sum(result * np.log(self.softmax_out))
        der_soft = self.softmax_out * (np.sum(self.softmax_out) - self.softmax_out) / np.sum(self.softmax_out) ** 2
        delta = cross_entropy * der_soft
        self.layers[-1].grad_vector(np.transpose(delta))
        out_with_bias = np.append([1], self.layers[-2].output_vector)
        out_with_bias = out_with_bias.reshape(len(out_with_bias), 1)
        der_loss = np.matmul(out_with_bias, self.layers[-1].grad_vect)
        self.layers[-1].loss_derivative(der_loss)
        self.layers[-1].weight_matrix_update(self.l_rate)

        for l in range(0, len(self.layers)-1, -1):
            grad = np.transpose(np.heaviside(self.layers[l].input_vector, 1)) * np.matmul(self.layers[l].weight_matrix, self.layers[l+1].grad_vect)
            der_loss = np.matmul(np.transpose(grad), self.layers[l-1].output_vector)
            self.layers[l].grad_vector(grad)
            self.layers[l].loss_derivative(der_loss)
            self.layers[l].weight_matrix_update(self.l_rate)

    def conv_error(self):
        error = self.layers[0].grad_vect
        return error


    def weight_decay(self):
        pass




