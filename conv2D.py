import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

class conv2D:

    def __init__(self, input, kernelNum, kernelSize, stride = 1):
        self.input = input
        self.err = np.array([])
        self.kernelNum = kernelNum
        self.kernelSize = kernelSize
        self.stride = stride
        self.bias = np.random.normal()
        self.kernelMatrix = np.random.rand(kernelSize,kernelSize,kernelNum)
        self.inputSize = input.shape()[0]
        self.convSize = (self.inputSize - kernelSize)//(stride-1) + 1
        self.outputSize = (self.convSize,self.convSize,kernelNum)

    def convolve(self):
        """Convolve RGB image with multiple kernels and sum for feature map"""
        convOut = np.empty((self.outputSize,self.outputSize,self.kernelNum)) # 3d array to store conv2d outputs
        for kernel_i in range(self.kernelNum): # goes through each 2d kernel
            temp = np.zeros((self.convSize,self.convSize)) # to sum channels into one
            for C in range(self.input.shape()[2]):
                for i in range(0,self.inputSize - np.ceil((self.kernelSize-1)/2),self.stride): #start top-left move until kernel hits edge
                    for j in range(0,self.inputSize - np.ceil((self.kernelSize-1)/2),self.stride):
                        temp += input[i:i+self.kernelSize,j:j+self.kernelSize,C]*self.kernelMatrix[:,:,kernel_i]
            convOut[:,:,kernel_i] = temp
        self.convOut = np.maximum(convOut + self.bias,np.zeros_like(convOut)) # apply ReLU before output

    def updateKernel(self):

