import numpy as np

class poolingLayer:
    def __init__(self, outputSize, ptype = 'max'):
        self.err = 0
        self.outputSize = outputSize
        self.ptype = ptype
        self.cellSize = None
        self.positions = []

    def pool(self, matrix):
        self.positions = []
        self.matrix = matrix
        self.cellSize = matrix.shape[0] // self.outputSize
        temp = np.empty((self.outputSize,self.outputSize, matrix.shape[2]))
        for kernel_i in np.arange(0, matrix.shape[2],dtype="int"):
            positions = []
            for i in np.arange(0, self.outputSize,dtype="int"):
                for j in np.arange(0,self.outputSize,dtype="int"):  # when we use the indices as earlier, the decimal parts cause problems
                    if self.ptype == 'max':
                        cell = matrix[i*self.cellSize:(i+1)*self.cellSize, j*self.cellSize:(j+1)*self.cellSize, kernel_i]
                        cellMax = np.max(cell)
                        temp[i, j, kernel_i] = cellMax
                        if cellMax == 0:
                            positions.append((0,0))
                        else:
                            ind = np.where(cell == cellMax) #keeps the info about where the maxes ar at the self.matrix
                            positions.append((ind[0][0], ind[1][0]))
                    if self.ptype == 'mean':
                        temp[i, j] = np.mean(matrix[i*self.cellSize:(i+1)*self.cellSize, j*self.cellSize:(j+1)*self.cellSize])
            positions = np.reshape(positions,(self.outputSize,self.outputSize,2))
            self.positions.append(positions)
        self.pooled = temp
        return self.pooled

    def getLoss(self, loss):
        lossMatSize = 2*np.sqrt(len(loss[0]))
        lossMatrix = np.reshape(loss, (len(self.positions),lossMatSize,lossMatSize))
        self.loss = np.zeros_like(lossMatrix)
        for kernel in self.positions:
            for i in np.arange(0, self.outputSize, dtype="int"):
                for j in np.arange(0, self.outputSize, dtype="int"):
                    self.loss[kernel, i*self.cellSize+self.positions[kernel][i,j,0],
                              j*self.cellSize+self.positions[kernel][i,j,1]] = lossMatrix[kernel,i,j]
        return self.loss