import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import math
class conv2D:
# input is now an attribiute of the function, so that the conv layer can be used with multiple images in train class
    def __init__(self,  kernelNum, kernelSize, stride = 1):
        #self.input = input
        self.err = np.array([])
        self.kernelNum = kernelNum
        self.kernelSize = kernelSize
        self.stride = stride
        self.bias = np.random.normal()
        self.kernelMatrix = np.random.rand(kernelSize,kernelSize,kernelNum)
        #self.inputSize = input.shape()[0]
        #self.convSize = (self.inputSize - kernelSize)//(stride-1) + 1
        self.outputSize = None # (self.convSize,self.convSize,kernelNum)

    def convolve(self,input):
        """Convolve RGB image with multiple kernels and sum for feature map"""
        inputSize = input.shape()[0]
        convSize = (inputSize - self.kernelSize) // (self.stride - 1) + 1
        self.outputSize = (convSize, convSize, self.kernelNum)
        convOut = np.empty((self.outputSize,self.outputSize,self.kernelNum)) # 3d array to store conv2d outputs
        for kernel_i in range(self.kernelNum): # goes through each 2d kernel
            temp = np.zeros((convSize,convSize)) # to sum channels into one
            for C in range(input.shape()[2]):
                for i in range(0,inputSize - np.ceil((self.kernelSize-1)/2),self.stride): #start top-left move until kernel hits edge
                    for j in range(0,inputSize - np.ceil((self.kernelSize-1)/2),self.stride):
                        temp += input[i:i+self.kernelSize,j:j+self.kernelSize,C]*self.kernelMatrix[:,:,kernel_i]
            convOut[:,:,kernel_i] = temp
        self.convOut = np.maximum(convOut + self.bias, np.zeros_like(convOut))  # apply ReLU before output

    def conv_backwards(self, loss_fc, positions):  # uses the loss of the first fully conected layer, to calculate its last layer
        mat_size = int(math.sqrt(np.size(loss_fc)))
        loss_fc=np.reshape(loss_fc, (mat_size, mat_size))
        self.kernelMatrix[:,:,-1]



    def updateKernel(self):
        pass

