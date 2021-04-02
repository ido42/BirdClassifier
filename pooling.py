import numpy as np

class poolingLayer:

    def __init__(self, matrix, outputSize, ptype = 'max'):
        self.err = 0
        self.matrix = matrix
        self.outputSize = outputSize
        self.ptype = ptype
        self.cellSize = matrix.shape()[0]//outputSize
        self.positions =[]

    def pool(self):
        temp = np.empty(self.outputSize,self.outputSize,self.matrix.shape()[2])
        for i in range(0,self.matrix.shape()[0],self.cellSize):
            for j in range(0,self.matrix.shape()[0],self.cellSize):
                if self.ptype == 'max':
                    temp[i/self.cellSize,j/self.cellSize] = np.max(self.matrix[i:i+self.cellSize, j:j+self.cellSize])

                if self.ptype == 'mean':
                    temp[i/self.cellSize,j/self.cellSize] = np.mean(self.matrix[i:i+self.cellSize, j:j+self.cellSize])
                mat = self.matrix[i:i + self.cellSize, j:j + self.cellSize]
                for i in range(len(mat[0])):
                    for j in range(len(mat[:, 0])):
                        if mat[i, j] == np.max(mat):
                            self.positions.append((i, j))
                            break
        self.pooled = temp

    def getErr(self):
        pass
