import os
import cv2
import numpy as np
from sklearn.model_selection import KFold

def image_load():
    # ProjectFolder = os.path.abspath(os.path.join(os.path.abspath(os.getcwd())))
    # directory_name = os.path.dirname
    # imgFolder = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), "Images"))
    trainPath = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train"))
    kf = KFold(n_splits=5, shuffle=True, random_state=1) # 5-fold validation split with random seed 1 for reproducibility
    birdList = os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train"))
    birdDict = {}
    birdsEncoded = {}
    birdsTrainFile = {}
    birdsTestFile = {}
    birdsTrain = {}
    birdsTest = {}

    for bird in birdList:
        birdDict[bird] = os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", bird)) # get images (str)
        oneHot = np.zeros(len(birdList))
        oneHot[birdList.index(bird)] = 1
        birdsEncoded[bird] = oneHot # one hot encode
        birdsTrainFile[bird] = [] # initializing dicts for k-fold split
        birdsTestFile[bird] = []
        birdsTrain[bird] = []
        birdsTest[bird] = []

    for bird in birdList: # generating the k-fold split
        for tempTrainInd, tempTestInd in kf.split(birdDict[bird]):
            for i in tempTrainInd:
                birdsTrainFile[bird].append(birdDict[bird][i])
                img = cv2.imread(trainPath.replace('\\', '/') + '/{0}/{1}'.format(bird,birdsTrainFile[bird][-1]))
                birdsTrain[bird].append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            for i in tempTestInd:
                birdsTestFile[bird].append(birdDict[bird][i])
                img = cv2.imread(trainPath.replace('\\', '/') + '/{0}/{1}'.format(bird, birdsTestFile[bird][-1]))
                birdsTest[bird].append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    return birdsEncoded, birdsTrain, birdsTest, birdsTrainFile, birdsTestFile
