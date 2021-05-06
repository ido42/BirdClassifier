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
        self.softmax_out = np.nan_to_num(np.exp(self.layers[-1].output_vector) / sum(np.exp(self.layers[-1].output_vector)))

    def back_prop_m(self, result):
        last_der_loss = np.matmul(self.layers[-1].input_vector.reshape(np.size(self.layers[-1].input_vector), 1),
                                  self.softmax_out.transpose() - result)
        self.layers[-1].grad_vector(-self.softmax_out.transpose() + result)
        self.layers[-1].loss_derivative(last_der_loss)
        self.layers[-1].weight_matrix = -self.layers[-1].loss_derivative_matrix * self.l_rate + self.layers[
            -1].weight_matrix
        for l in range(len(self.layers) - 1, 0, -1):  # other layers
            der_relu = np.heaviside(self.layers[l - 1].output_vector, 0)
            delta = np.matmul(self.layers[l].weight_matrix[1:], self.layers[l].grad_vect.transpose()) * der_relu
            self.layers[l - 1].grad_vector(delta.transpose())
            der_loss = np.matmul(self.layers[l - 1].input_vector.reshape(np.size(self.layers[l - 1].input_vector), 1),
                                 self.layers[l - 1].grad_vect)
            self.layers[l - 1].loss_derivative(der_loss)
            self.layers[l - 1].weight_matrix = -self.layers[l - 1].loss_derivative_matrix * self.l_rate + self.layers[
                l - 1].weight_matrix


    def dropout(self, drop_probability):
        for d in range(len(self.layers)):
            drop_neurons = np.random.binomial(1, 1 - drop_probability, len(self.layers[d].input_vector))
            self.layers[d].input_vector = self.layers[d].input_vector * drop_neurons