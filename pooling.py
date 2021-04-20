import numpy as np

class poolingLayer:

    def __init__(self, outputSize, ptype = 'max'):
        self.err = 0
        self.outputSize = outputSize
        self.ptype = ptype
        self.cellSize = None
        self.positions = []

    def pool(self, matrix):
        self.matrix = matrix
        self.cellSize = matrix.shape()[0] // self.outputSize
        temp = np.empty(self.outputSize,self.outputSize,matrix.shape()[2])
        for i in np.arange(0, self.outputSize,dtype="int"):
            for j in np.arange(0,self.outputSize,dtype="int"):  # when we use the indices as earlier, the decimal parts cause problems
                if self.ptype == 'max':
                    cellMax = np.max(matrix[i*self.cellSize:(i+1)*self.cellSize, j*self.cellSize:(j+1)*self.cellSize])
                    temp[i, j] = cellMax
                    ind = np.where(self.matrix == cellMax) #keeps the info about where the maxes ar at the self.matrix
                    self.positions.append(([ind[0][0], ind[1][0]]))
                if self.ptype == 'mean':
                    temp[i,j] = np.mean(matrix[i*self.cellSize:(i+1)*self.cellSize, j*self.cellSize:(j+1)*self.cellSize])
        self.pooled = temp

    def getLoss(self, loss):
        lossMatSize = 2*np.sqrt(len(loss[0]))
        lossMatrix = np.reshape(loss,self.matrix.shape())
        self.loss = np.zeros_like(lossMatrix)
        for pos in self.positions:
            self.loss[pos] = lossMatrix[pos]
        return self.loss