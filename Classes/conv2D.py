import numpy as np
from scipy.signal import convolve2d

class conv2D:


    def __init__(self,  kernelNum, kernelSize, inputDepth, learningRate, stride = 1):
        self.learningRate = learningRate
        self.kernelNum = kernelNum
        self.kernelSize = kernelSize
        self.stride = stride
        self.bias = np.random.uniform(-1,1,(kernelNum,1))
        self.kernelMatrix = np.random.uniform(-1,1,(kernelNum,kernelSize,kernelSize,inputDepth))


    def forward(self, input): # convolution
        self.input = input
        self.inputSize = input.shape
        convSize = (self.inputSize[0] - self.kernelSize) // self.stride + 1
        self.outputSize = (convSize, convSize, self.kernelNum)
        convOut = np.zeros(self.outputSize)  # 3d array to store conv2d outputs
        for f in np.arange(self.kernelNum, dtype="int"):
            for C in np.arange(self.inputSize[2], dtype="int"):
                convOut[:, :, f] += convolve2d(input[:,:,C],self.kernelMatrix[f, :, :, C], mode='valid') + self.bias[f]
            self.convOut = np.maximum(convOut, 0)  # apply ReLU before output
        return self.convOut


    def backward(self, grad):
        gradReLU = grad * np.heaviside(self.convOut, 1)
        # bias gradient
        self.gradBias = np.sum(gradReLU, axis=(0,1))
        # weight gradient
        self.gradKernels = np.zeros_like(self.kernelMatrix)
        for f in np.arange(0, self.kernelNum, dtype="int"):
            for c in np.arange(0, self.inputSize[2], dtype="int"):
                self.gradKernels[f,:,:,c] = convolve2d(gradReLU[:,:,f],self.input[:,:,c], mode='valid')
        # propagation gradient
        self.grad = np.zeros(self.inputSize)
        for c in np.arange(0, self.inputSize[2], dtype="int"):
            for f in np.arange(0, self.kernelNum, dtype="int"):
                self.grad[:,:,c] += convolve2d(np.rot90(self.kernelMatrix[f,:,:,c], k=2), gradReLU[:,:,f])
        # updating the parameters
        self.kernelMatrix = self.kernelMatrix - self.learningRate*self.gradKernels
        self.bias = self.bias - self.learningRate*self.gradBias