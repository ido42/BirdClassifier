import os
import numpy as np
import cv2
import random

ProjectFolder = os.path.abspath(os.path.join(os.path.abspath(os.getcwd())))
directory_name = os.path.dirname
imgFolder = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), "Images"))
imgTrain = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train"))
imgTrainBO = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "BARN OWL"))
imgTrainF = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "FLAMINGO"))
imgTrainTM = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "TURQUOISE MOTMOT"))

birdList = os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train"))
birdDict = {"FLAMINGO": os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "FLAMINGO")),
            "BARN OWL": os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "BARN OWL")),
            "TURQUOISE MOTMOT": os.listdir(os.path.join(os.path.abspath(os.getcwd()), "Images", "Train", "TURQUOISE MOTMOT"))}

birdsEncoded = {"FLAMINGO": [0, 0, 1], "BARN OWL": [0, 1, 0], "TURQUOISE MOTMOT": [0,0,1]}
imageNum = 0
for bird in birdList:
    imageNum += len(birdDict[bird])
random.shuffle(birdList)
train_birds = []
train_species = []
valid_birds = []
valid_species = [] # species in encoded form ,list
all_birds = [] # strings in the form "099.jpg"
all_species = [] # strings, names of the corresponding birds
for i in range(imageNum):
    randBird = random.choice(birdList)
    randImg = random.choice(birdDict[randBird])
    if i > 4 * imageNum // 5:
        train_birds.append(randImg)
        train_species.append(birdsEncoded[randBird])

    else:
        valid_species.append(birdsEncoded[randBird])
        valid_birds.append(randImg)
    all_birds.append(randImg)
    all_species.append(randBird)

