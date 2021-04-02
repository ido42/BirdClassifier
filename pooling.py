import numpy as np

class poolingLayer:

    def __init__(self, outputSize, ptype = 'max'):
        self.err = 0
        self.outputSize = outputSize
        self.ptype = ptype
        self.cellSize = matrix.shape()[0]//outputSize
        self.positions = []

    def pool(self, matrix):
        self.matrix = matrix
        temp = np.empty(self.outputSize,self.outputSize,matrix.shape()[2])
        for i in range(0,matrix.shape()[0],self.cellSize):
            for j in range(0,matrix.shape()[0],self.cellSize):
                if self.ptype == 'max':
                    cellMax = np.max(matrix[i:i + self.cellSize, j:j + self.cellSize])
                    temp[i/self.cellSize,j/self.cellSize] = cellMax
                    for index0 in np.where(matrix == cellMax)[0]:
                        posSetflag = 0
                        for index1 in np.where(matrix == cellMax)[1]:
                            if index0 < i or index1 < j:
                                break
                            else:
                                posSetflag = 1
                                self.positions.append((index0,index1))
                                break
                        if posSetflag:
                            break
                if self.ptype == 'mean':
                    temp[i/self.cellSize,j/self.cellSize] = np.mean(matrix[i:i+self.cellSize, j:j+self.cellSize])
        self.pooled = temp

    def getErr(self):
        pass
