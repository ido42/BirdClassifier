import os
import cv2
import numpy as np
from sklearn.model_selection import KFold
import random
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

def shuffled_matrices(images,encoded_birds,num_classes,shuffle=True):

    shuffled_img_mat=np.zeros((1,50176)) #the matrix containing all images, each flatten image in a row
    labels_mat=np.zeros((1,num_classes)) # the matrix containing labels for the corresponding images, rows are encoded bird keys
    bird_names=list(images.keys())
    count=0
    for bird in range(len(bird_names)):
        for i in range(len(images[bird_names[bird]])):
            shuffled_img_mat=np.append(shuffled_img_mat,np.reshape(images[bird_names[bird]][i],(1,np.size(images[bird_names[bird]][i]))))
            labels_mat=np.append(labels_mat,encoded_birds[bird_names[bird]])
            count +=1
    labels_mat=np.reshape(labels_mat[num_classes:],(count,num_classes))
    shuffled_img_mat=np.reshape(shuffled_img_mat[50176:],(count,50176))
    if shuffle == True: #if false an ordered array
        for s in range(count//2):
            location=random.randint(0,count-1)
            destination=random.randint(0,count-1)
            shuffled_img_mat_copy=shuffled_img_mat[destination]
            labels_mat_copy=labels_mat[destination]
            shuffled_img_mat[destination]=shuffled_img_mat[location]
            labels_mat[destination]=labels_mat[location]
            shuffled_img_mat[location]=shuffled_img_mat_copy
            labels_mat[location]=labels_mat_copy
    return shuffled_img_mat,labels_mat



