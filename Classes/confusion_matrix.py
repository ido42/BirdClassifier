import numpy as np


class confusion_mat():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.c_mat = np.zeros((self.num_classes, self.num_classes))
        self.fail=0
    def update(self, true_class, predicted_class,Failure):

        if Failure==True:
            self.fail+=1
        else:
            t = int(np.where(true_class == np.max(true_class))[0])
            p = int(np.where(predicted_class == np.max(predicted_class))[0])
            self.c_mat[p, t] += 1
        return self.c_mat,self.fail
