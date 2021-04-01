from Classes.Layer import*


class ANN():
    def __init__(self,num_layer,learning_rate,layers_neurons): #layers_neurons is a list
        self.num_layer = num_layer
        self.l_rate = learning_rate
        self.layers = []
        for l in range(num_layer):
            self.layers.append(Layer(layers_neurons[l],layers_neurons[l+1]))

    def forward_pass(self, flat_image):  # takes flat image calculates the output once, using the current weights
        self.layers[1].take_input(flat_image)

        for i in range(1, len(self.layers)):
            sigm_out = 1/(np.exp(-self.layers[i-1].layer_output()))
            self.layers[i].take_input(sigm_out)
        self.layers[-1].take_input(1/(np.exp(-self.layers[-2].layer_output())))

    def back_prop(self, result): # result is in the form of a vector which is the same size with output
        pass
        #dLoss =
        #self.layers
        #for l in range(0, len(self.layers), -1): # backwards
         #   dLoss=



    def train(self, flat_image, result):  # takes one flatten image, updates weights using forward_pass and back_prop
        pass

#    def add_layer(self, num_new_layer):
 #       for layer in range(num_new_layer):
  #          self.layers = self.layers.append(Layer(sel))
   #     self.num_layer += num_new_layer

    #def remove_layer(self, num_remove_layer):
     #   for layer in range(num_remove_layer):
      #      self.layers = self.layers.append(Layer(self.inputs, self.weights))
       # self.num_layer -= num_remove_layer