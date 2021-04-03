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
        #self.input = None
        self.outputSize = 0  # (self.convSize,self.convSize,kernelNum)


    def convolve(self, input_img):
        """Convolve RGB image with multiple kernels and sum for feature map"""
        #self.input = input_img
        inputSize = input_img.shape[0]#len(input_img[0][0])
        convSize = (inputSize - self.kernelSize) // (self.stride) + 1
        self.outputSize = (convSize, convSize, self.kernelNum)
        convOut = np.empty(self.outputSize)  # 3d array to store conv2d outputs
        for kernel_i in np.arange(self.kernelNum, dtype="int"):
            temp = np.zeros((convSize,convSize))
            for C in np.arange(input_img.shape[2], dtype="int"):
                for i in np.arange(0, inputSize - np.ceil((self.kernelSize - 1) / 2), self.stride, dtype="int"):  # start top-left move until kernel hits edge

                    for j in np.arange(0, inputSize - np.ceil((self.kernelSize - 1) / 2), self.stride, dtype="int"):
                        temp[i:i + self.kernelSize, j:j + self.kernelSize]  += input_img[i:i + self.kernelSize, j:j + self.kernelSize, C]* self.kernelMatrix[:, :, kernel_i]


                convOut[:, :, kernel_i] = temp
                self.convOut = np.maximum(convOut + self.bias, np.zeros_like(convOut))  # apply ReLU before output

