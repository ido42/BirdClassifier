import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

class conv2D:

    def __init__(self, image, kernelNum, kernelSize, stride = 1):
        self.kernelMatrix = np.random.rand(kernelSize,kernelSize,kernelNum)
        self.imageSize = image.shape()[0]
        self.convSize = (self.imageSize - kernelSize)//(stride-1) + 1
        self.outputSize = (self.convSize,self.convSize,kernelNum)

    def convolve(self,image,kernelNum,kernelSize,stride):
        """Convolve RGB image with multiple kernels and sum for feature map"""
        convOut = np.zeros((self.outputSize,self.outputSize,kernelNum))
        for kernel_i in range(kernelNum): # goes through each 2d kernel
            temp = np.zeros((self.convSize,self.convSize))
            for C in range(3):  # goes through each channel
                for i in range(0,self.imageSize - np.ceil((kernelSize-1)/2),stride): #start top-left move until kernel hits edge
                    for j in range(0,self.imageSize - np.ceil((kernelSize-1)/2),stride):
                        temp += image[i:i+kernelSize,j:j+kernelSize,C]*self.kernelMatrix[:,:,kernel_i]
            convOut[:,:,kernel_i] = temp
        return convOut