import numpy as np

class Layer():
    def __init__(self, begin_neuron_count, end_neuron_count, input_v=None):
        self.begin_neuron_count = begin_neuron_count
        self.end_neuron_count = end_neuron_count
        if input_v!=None:
            self.input_vector = np.array([1], input_v) #1 is because of mean term
        else:
            self.input_vector= None
        self.weight_matrix = np.random.rand(self.begin_neuron_count + 1, self.end_neuron_count)
        self.loss_derivative_matrix = np.zeros(np.shape(self.weight_matrix ))
        self.grad_vect = np.zeros((self.end_neuron_count,1))
        if self.input_vector != None:
           self.output_vector = np.matmul(self.input_vector, self.weight_matrix)
        else:
            self.output_vector=None

    def take_input(self, input_v):
        self.input_vector = np.append([1], input_v)

    def weight_matrix_update(self, l_rate):  # put the weights
        #self.weight_matrix -= l_rate*self.loss_derivative_matrix
        self.weight_matrix = self.weight_matrix-self.loss_derivative_matrix*l_rate
    def layer_output(self):
        self.output_vector = np.matmul(self.input_vector, self.weight_matrix)
        self.output_vector=self.output_vector.reshape(len(self.output_vector),1)

    def loss_derivative(self, mat):
        self.loss_derivative_matrix = mat

    def grad_vector(self,vect):
        self.grad_vect = vect