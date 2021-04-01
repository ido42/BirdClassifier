import numpy as np

class poolingLayer:

    def __init__(self, matrix, outputSize, ptype = 'max'):
        self.cellSize = matrix.shape()[0]//outputSize

    def pool(self, matrix, outputSize, ptype):
        temp = np.empty(outputSize,outputSize,matrix.shape()[2])
        for i in range(0,matrix.shape()[0],self.cellSize):
            for j in range(0,matrix.shape()[0],self.cellSize):
                if ptype == 'max':
                    temp[i/self.cellSize,j/self.cellSize] = np.max(matrix[i:i+self.cellSize, j:j+self.cellSize])
                if ptype == 'mean':
                    temp[i/self.cellSize,j/self.cellSize] = np.mean(matrix[i:i+self.cellSize, j:j+self.cellSize])
        return temp
