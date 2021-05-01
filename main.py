import cv2
import numpy as np
from conv2D import *
from pooling import *
from matplotlib import pyplot as plt
from Classes.ANN import *

#from Train import *
from image_load import *

"""Trainer = Train(3, 5, 25, 5, 0.1, [5, 5, 5, 2], 1, 'max')

for bird in range(len(train_species)):
    if train_species[bird] == [0, 1]:
        img = cv2.imread(imgTrainF+"\\"+train_birds[bird])
    else:
        img = cv2.imread(imgTrainBO+"\\"+train_birds[bird])
    Trainer.train_with_one_img(img, train_species[bird] )"""

x = np.array([0,1,2,2,3,4])
num_layer=4
learning_rate=0.01
layers_neurons=[6,2,3,6,3]
network=ANN(num_layer, learning_rate, layers_neurons)
result=np.array([0,0,1])
soft_list=[]

for i in range(1000):
    network.forward_pass(x)
    print(network.softmax_out)
    network.back_prop(result)
    soft_list.append(network.softmax_out)