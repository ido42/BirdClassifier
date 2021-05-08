from  Classes.Layer import *
import numpy as np
import cupy as cp

class ANN:


    def __init__(self, l_rate, neurons):  # layers_neurons is a list
        self.loss = []
        self.num_layer = len(neurons) - 1
        self.l_rate = l_rate
        self.layers = []
        for l in range(self.num_layer - 1):
            self.layers.append(Layer(neurons[l], neurons[l + 1]))
        self.layers.append(Layer(neurons[-2], neurons[-1], activation='softmax'))


    def forward_pass(self, flat_image):  # takes flat image calculates the output once, using the current weights
        self.layers[0].forward(flat_image)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1].output_vector)
        self.out = self.layers[-1].output_vector


    def back_prop_m(self, labels):
        initGrad = self.layers[-1].output_vector - labels
        # initGradW = cp.matmul(initGrad, self.layers[-1].input_vector.transpose())
        self.layers[-1].backward(initGrad, self.l_rate)
        for l in range(len(self.layers)-2, -1, -1):  # other layers
            gradRelu = self.layers[l+1].grad * np.heaviside(self.layers[l].output_vector, 0)
            self.layers[l].backward(gradRelu, self.l_rate)
        # self.loss = cp.sum(- labels * cp.log(self.out, where=self.out!=0))
        self.loss.append(np.sum((self.preventOF(self.out) - labels)**2))


    def dropout(self, drop_probability):
        drop_neurons = cp.random.binomial(1, 1 - drop_probability, len(self.layers[0].input_vector))
        self.layers[0].input_vector = self.layers[0].input_vector * drop_neurons
        for d in range(0, len(self.layers)):
            drop_neurons = cp.random.binomial(1, 1 - drop_probability, (len(self.layers[d].output_vector),1))
            self.layers[d].output_vector = self.layers[d].output_vector * drop_neurons


    @staticmethod
    def preventOF(mat):  # to prevent overflows
        temp = np.where(mat < 1e-200, 0, mat)
        temp = np.where(0 < mat, temp, mat)
        return np.where(1e200 < temp, 1e200, temp)