from  Classes.Layer import *
import numpy as np

class ANN:
    def __init__(self, learning_rate, layers_neurons):  # layers_neurons is a list
        self.num_layer = len(layers_neurons) - 1
        self.l_rate = learning_rate
        self.layers = []
        for l in range(self.num_layer):
            self.layers.append(Layer(layers_neurons[l], layers_neurons[l + 1]))
        self.softmax_out = None

    def forward_pass(self, flat_image):  # takes flat image calculates the output once, using the current weights

        self.layers[0].take_input(flat_image)
        self.layers[0].layer_output()
        for i in range(1, len(self.layers)):
            relu_out = np.maximum(0, self.layers[i - 1].output_vector)
            self.layers[i].take_input(relu_out)
            self.layers[i].layer_output()
        self.softmax_out = np.exp(self.layers[-1].output_vector) / sum(np.exp(self.layers[-1].output_vector))

    def back_prop(self, result):  # result is in the form of a vector which is the same size with output
        cross_entropy = -np.sum(result * np.log(self.softmax_out))  # cross-entropy cost function
        der_soft = np.exp(self.layers[-1].output_vector) * (
                    np.sum(np.exp(self.layers[-1].output_vector)) - np.exp(self.layers[-1].output_vector)) / np.sum(
            np.exp(self.layers[-1].output_vector)) ** 2
        delta = cross_entropy * der_soft
        self.layers[-1].grad_vector(np.transpose(delta))
        der_loss = np.matmul(self.layers[-1].input_vector.reshape(len(self.layers[-1].input_vector), 1),
                             self.layers[-1].grad_vect)
        self.layers[-1].loss_derivative(der_loss)
        self.layers[-1].weight_matrix_update(self.l_rate)  # until here the last layer

        for l in range(len(self.layers) - 1, 0, -1):
            der_relu = np.heaviside(self.layers[l - 1].output_vector, 1)
            delta = np.zeros((1, len(self.layers[l - 1].output_vector)))
            for i in range(len(self.layers[l - 1].output_vector)):
                for j in range(len(self.layers[l].output_vector)):
                    delta[0][i] += self.layers[l].grad_vect[0][j] * self.layers[l].weight_matrix[i + 1][j] * \
                                   der_relu[i]
            self.layers[l - 1].grad_vector(delta)
            der_loss = np.matmul(self.layers[l - 1].input_vector.reshape(len(self.layers[l - 1].input_vector), 1),
                                 self.layers[l - 1].grad_vect)
            self.layers[l - 1].loss_derivative(der_loss)
            self.layers[l - 1].weight_matrix_update(self.l_rate)