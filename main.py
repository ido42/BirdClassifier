import cv2
import numpy as np
from conv2D import *
from pooling import *
import sys
from matplotlib import pyplot as plt
from Classes.ANN import *
from logistic_regression import *
#from Train import *
from image_load import *




"""test for trainer class
Trainer = Train(3, 5, 25, 5, 0.1, [5, 5, 5, 2], 1, 'max')

for bird in range(len(train_species)):
    if train_species[bird] == [0, 1]:
        img = cv2.imread(imgTrainF+"\\"+train_birds[bird])
    else:
        img = cv2.imread(imgTrainBO+"\\"+train_birds[bird])
    Trainer.train_with_one_img(img, train_species[bird] )"""

#test for ANN
tests = 500
count = 0
for j in range(tests):
    sys.stdout.write("\r----- %d%% done -----" % (100*(j+1)/tests))
    sys.stdout.flush()

#   parameters
    x = np.array(range(1000))
    x = (x - x.mean())/x.var()
    learning_rate=0.01
    layers_neurons=[1000,100,7]
    network=ANN(learning_rate, layers_neurons)
    result=np.array([0,0,1,0,0,0,0])
    soft_list=[]

    for i in range(500):
        network.forward_pass(x)
        # print(network.softmax_out)
        res = True in [ele < 10e-15 or np.isnan(ele) for ele in network.softmax_out] # if a value is dangerously small or NaN
        if res:
            # print('Epoch', i + 1, 'overflows.')
            # if 1-network.softmax_out[2] < 10e-90:
                # print('Successfully converged.')
            break
        network.back_prop(result)
        soft_list.append(network.softmax_out)

    if 1 - network.softmax_out[2] < 0.05: # if the output is sufficiently close to the result
        count += 1
        # print('Successfully converged.')
    # else:
        # print('Failed to converge.')"""
print('\n'+str(100*count/tests)+'% accuracy.')