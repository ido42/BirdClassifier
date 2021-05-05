import numpy as np
from scipy.signal import convolve2d

class conv2D:


    def __init__(self,  kernelNum, kernelSize, stride = 1):
        self.kernelNum = kernelNum
        self.kernelSize = kernelSize
        self.stride = stride
        self.bias = np.random.uniform(-1,1,(kernelNum,1))
        self.kernelMatrix = np.random.uniform(-1,1,(kernelNum,kernelSize,kernelSize))


    def forward(self, input): # convolution
        self.input = input
        self.inputSize = input.shape
        convSize = (self.inputSize[0] - self.kernelSize) // self.stride + 1
        self.outputSize = (convSize, convSize, self.kernelNum)
        convOut = np.empty(self.outputSize)  # 3d array to store conv2d outputs
        for kernel_i in np.arange(self.kernelNum, dtype="int"):
            temp = np.zeros((convSize,convSize))
            for C in np.arange(self.inputSize[1], dtype="int"):
                for i in np.arange(0, self.inputSize[0] - self.kernelSize + 1, self.stride, dtype="int"):  # start top-left move until kernel hits edge
                    for j in np.arange(0, self.inputSize[0] - self.kernelSize + 1, self.stride, dtype="int"):
                        temp[i, j] = np.sum(input[i:i + self.kernelSize, j:j + self.kernelSize, C] * self.kernelMatrix[kernel_i, :, :]) + self.bias[kernel_i]
                convOut[:, :, kernel_i] += temp
            self.convOut = np.maximum(convOut, np.zeros_like(convOut))  # apply ReLU before output
        return self.convOut


    def getGrad(self, grad):
        # bias gradient
        gradReLU = grad * np.heaviside(self.convOut, 1)
        self.gradBias = np.sum(gradReLU, axis=(0,1))
        # weight gradient
        temp = np.empty((self.kernelSize,self.kernelSize,self.kernelNum,self.inputSize[1]))
        self.gradWeight = np.empty((self.kernelSize,self.kernelSize,self.kernelNum))
        for f in np.arange(0, self.kernelNum, dtype="int"):
            for c in np.arange(0, self.inputSize[1], dtype="int"):
                temp[:,:,f,c] = convolve2d(np.rot90(gradReLU[:,:,f], k=2),self.input[:,:,c])
            self.gradWeight[:,:,f] = np.mean(temp, axis=3)
        # # propagation gradient
        # temp = np.empty(self.inputSize)
        # for f in np.arange(0, self.kernelNum, dtype="int"):
        #
        # self.grad
