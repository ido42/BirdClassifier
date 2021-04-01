import os
import numpy as np
import cv2

ProjectFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
directory_name = os.path.dirname
imgFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Images"))
imgTrain = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Images", "Train"))


birdList = os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train"))
birdDict = {"FLAMINGO":os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "FLAMINGO")),
            "BARN OWL":os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "BARN OWL"))}
imageNum = 0
for bird in birdList:
    imageNum += len(birdDict[bird])

np.random.Generator.shuffle(birdList)
for i in range(imageNum):
    if i > imageNum/5:
        randBird = np.random.Generator.choice(birdList)
        randImg = np.random.Generator.choice(birdDict[randBird])
        trainSet
    else:
        testSet


