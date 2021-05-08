from time import time
import pickle
import sys
from Train import *
from image_load import *
from Classes.confusion_matrix import *



rng = np.random.default_rng()
labels, picturesTrain, picturesTest, _, _ = image_load()
subsample = poolingLayer(56)
ann = ANN(0.001, (3136,10))
confTrain = confusion_mat(10)
confTest = confusion_mat(10)

# t1 = time()
for epoch in range(5):
    t2 = time()
    countBird = 0
    for bird in rng.permutation(list(labels)):
        t3 = time()
        count = 0
        for picture in picturesTrain[bird][epoch]:
            sys.stdout.write("\r----- Bird "+str(countBird+1)+":"+bird+" {:3.4}% done -----".format(100 * count / len(picturesTrain[bird][epoch])))
            sys.stdout.flush()
            t4 = time()
            picSub = subsample.pool(picture).reshape(1, 56*56)
            ann.forward_pass(picSub)
            ann.back_prop_m(labels[bird].reshape(1,10))
            d4 = time() - t4
            count += 1
            # if np.where(ann.out == ann.out.max())[0][0] == np.where(labels[bird] == 1)[0][0]:
            confTrain.update(labels[bird].reshape(10,1),ann.out)
            epoch
        countBird += 1
        # d3 = time() - t3
        # epoch
    print("\nEpoch:", epoch, "Error:", np.mean(ann.loss), "\n"+str(time() - t2))
    print(confTrain.c_mat)
    ann.loss = []

    for bird in rng.permutation(list(labels)):
        for picture in picturesTest[bird][epoch]:
            t8 = time()
            picSub = subsample.pool(picture).reshape(1,56)
            ann.forward_pass(picSub)
            # if np.where(ann.out == ann.out.max())[0][0] == np.where(labels[bird] == 1)[0][0]:
            confTrain.update(labels[bird].reshape(10,1),ann.out)
            d8 = time() - t8
    print(confTest.c_mat)
    # print("\n",count / np.sum([len(picturesTest[i][0]) for i in list(labels)]) * 100, '%')
    # epoch
# d1 = time() - t1