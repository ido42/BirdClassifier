import os
import cv2
from sklearn.model_selection import KFold
import random
from Classes.pooling import *
def image_load(BW=True):
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
                if BW:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img / 255
                    birdsTrain[bird].append(img)
                else:
                    img = img / 255
                    birdsTrain[bird].append(img)
            for i in tempTestInd:
                birdsTestFile[bird].append(birdDict[bird][i])
                img = cv2.imread(trainPath.replace('\\', '/') + '/{0}/{1}'.format(bird, birdsTestFile[bird][-1]))
                if BW:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img / 255
                    birdsTest[bird].append(img)
                else:
                    img = img / 255
                    birdsTest[bird].append(img)

    return birdsEncoded, birdsTrain, birdsTest, birdsTrainFile, birdsTestFile

def img_load_matrices(pool_size,images,encoded_birds,num_classes):
    pool=poolingLayer(pool_size)
    inp_img_mat=np.zeros((1,pool_size**2)) #the matrix containing all images, each flatten image in a row
    labels_mat=np.zeros((1,num_classes)) # the matrix containing labels for the corresponding images, rows are encoded bird keys
    bird_names=list(images.keys())
    count=0
    for bird in range(len(bird_names)):
        for i in range(len(images[bird_names[bird]])):
            inp=pool.pool(images[bird_names[bird]][i])
            inp_img_mat=np.append(inp_img_mat,np.reshape(inp,(1,np.size(inp))))
            labels_mat=np.append(labels_mat,encoded_birds[bird_names[bird]])
            count +=1
    labels_mat=np.reshape(labels_mat[num_classes:],(count,num_classes))
    inp_img_mat=np.reshape(inp_img_mat[pool_size**2:], (count, pool_size**2))
    inp_img_mat = (inp_img_mat - inp_img_mat.mean()) / np.sqrt(inp_img_mat.var())
    return inp_img_mat, labels_mat


def shuffle_matrix(inp_img_mat, labels_mat):
    count=np.shape(inp_img_mat)[0]
    for s in range(count//2):
        location=random.randint(0,count-1)
        destination=random.randint(0,count-1)
        shuffled_img_mat_copy=inp_img_mat[destination]
        sh_labels_mat_copy=labels_mat[destination]
        inp_img_mat[destination]=inp_img_mat[location]
        labels_mat[destination]=labels_mat[location]
        inp_img_mat[location]=shuffled_img_mat_copy
        labels_mat[location]=sh_labels_mat_copy
    return inp_img_mat,labels_mat



