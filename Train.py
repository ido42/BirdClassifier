from conv2D import *
from pooling import *
from Classes.ANN import *
class Train():
    def __init(self, convKernelNum, convKernelSize, poolOutputSize,
               ann_num_layer, ann_learning_rate, ann_layers_neurons, conv_stride = 1, pool_ptype = 'max'):
        self.convKernelNum = convKernelNum
        self.convKernelSize = convKernelSize
        self.conv_stride = conv_stride
        self.poolOutputSize = poolOutputSize
        self.pool_ptype = pool_ptype
        self.ann_num_layer = ann_num_layer
        self.ann_learning_rate = ann_learning_rate,
        self.ann_layers_neurons = ann_layers_neurons
        self.conv = conv2D(convKernelNum, convKernelSize, conv_stride)
        self.prediction = None


    def train_with_one_img(self, input, input_label):
        while input_label != self.prediction:
            conved_img = self.conv.convolve(input)
            pool = poolingLayer(conved_img, self.poolOutputSize, self.pool_ptype)
            pooled = pool.pooled
            flatten = np.reshape(pooled(1, np.size(pooled)))
            fully_connected = ANN(self.ann_num_layer, self.ann_learning_rate, self.ann_layers_neurons)
            fully_connected.forward_pass(flatten)
            self.prediction = fully_connected.layers[-1].output_vector
            fully_connected.back_prop(input_label)
            #self.conv.conv_backwards(error, pooled.positions)

