import numpy as np

class poolingLayer:
    def __init__(self, outputSize):
        self.outputSize = outputSize
        self.cellSize = None
        self.positions = []

    def pool(self, matrix):
        self.positions = np.zeros_like(matrix)
        try:
            self.kernelNum = matrix.shape[2]
        except:
            self.kernelNum = 1
        self.cellSize = matrix.shape[0] // self.outputSize
        temp = np.zeros((self.outputSize,self.outputSize, self.kernelNum))
        if self.kernelNum==1:
            for i in np.arange(0, self.outputSize,dtype="int"):
                for j in np.arange(0,self.outputSize,dtype="int"):
                    cell = matrix[i*self.cellSize:(i+1)*self.cellSize, j*self.cellSize:(j+1)*self.cellSize]
                    cellMax = np.max(cell)
                    temp[i, j] = cellMax
                    ind = np.where(cell == cellMax)
                    if cellMax == 0:
                        self.positions[i*self.cellSize, j*self.cellSize] = 1
                    else:
                        self.positions[i*self.cellSize+ind[0][0], j*self.cellSize+ind[1][0]] = 1
        else:
            for kernel_i in np.arange(0, self.kernelNum,dtype="int"):
                for i in np.arange(0, self.outputSize,dtype="int"):
                    for j in np.arange(0,self.outputSize,dtype="int"):
                        cell = matrix[i*self.cellSize:(i+1)*self.cellSize, j*self.cellSize:(j+1)*self.cellSize, kernel_i]
                        cellMax = np.max(cell)
                        temp[i, j, kernel_i] = cellMax
                        ind = np.where(cell == cellMax)
                        if cellMax == 0:
                            self.positions[i*self.cellSize, j*self.cellSize, kernel_i] = 1
                        else:
                            self.positions[i*self.cellSize+ind[0][0], j*self.cellSize+ind[1][0], kernel_i] = 1
        self.pooled = temp
        return self.pooled

    def getGrad(self, grad):
        if grad.shape[1] == 1:
            gradMatSize = np.int_(np.sqrt(grad.shape[0]/self.kernelNum))
            gradMatrix = np.reshape(grad, (self.kernelNum, gradMatSize,gradMatSize))
            gradMatrix = gradMatrix.repeat(self.cellSize, axis=1).repeat(self.cellSize, axis=2)
            self.grad = np.moveaxis(self.positions, -1, 0)*gradMatrix
            self.grad = np.moveaxis(self.grad, 0, -1)
        else:
            gradMatrix = gradMatrix.repeat(self.cellSize, axis=0).repeat(self.cellSize, axis=1)
            self.grad = self.positions * gradMatrix