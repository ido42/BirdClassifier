from time import time
import sys
from Train import *
from image_load import *


t = time()
labels, picturesTrain, picturesTest, _, _ = image_load(BW=False)
rng = np.random.default_rng()
ann = ANN(0.0001, (5*5*30,3))
conv1 = conv2D(5,5,3,0.0001)
conv2 = conv2D(10,3,5,0.0001)
conv3 = conv2D(15,3,10,0.0001)
conv4 = conv2D(30,3,15,0.0001)
pool1 = poolingLayer(110)
pool2 = poolingLayer(36)
pool3 = poolingLayer(17)
pool4 = poolingLayer(5)
d = time() - t

# t1 = time()
for epoch in range(5):
    t2 = time()
    for bird in rng.permutation(list(labels)):
        count = 0
        for picture in picturesTrain[bird][epoch]:
            sys.stdout.write("\r----- "+bird+" {:3.4}% done -----".format(100 * count / len(picturesTrain[bird][epoch])))
            sys.stdout.flush()
            if count == 2:
                break
            t3 = time()
            x1 = conv1.forward(picture)
            y1 = pool1.pool(x1)
            x2 = conv2.forward(y1)
            y2 = pool2.pool(x2)
            x3 = conv3.forward(y2)
            y3 = pool3.pool(x3)
            x4 = conv4.forward(y3)
            y4 = pool4.pool(x4)
            ann.forward_pass(y4.reshape(5*5*30,1))
            ann.back_prop_m(labels[bird].reshape(10,1))
            pool4.getGrad(ann.layers[-1].grad)
            conv4.backward(pool4.grad)
            pool3.getGrad(conv4.grad)
            conv3.backward(pool3.grad)
            pool2.getGrad(conv3.grad)
            conv2.backward(pool2.grad)
            pool1.getGrad(conv2.grad)
            conv1.backward(pool1.grad)
            d3 = time() - t3
            count += 1

            cv2.imshow('Conv1 Output (220x220)', x1[:, :, 0] / np.max(x1[:, :, 0]))
            cv2.imshow('Pool1 Output (110x110)', y1[:, :, 0] / np.max(y1[:, :, 0]))
            cv2.imshow('Conv2 Output (108x108)', x1[:, :, 0] / np.max(x1[:, :, 0]))
            cv2.imshow('Pool2 Output (36x36)', y1[:, :, 0] / np.max(y1[:, :, 0]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(epoch)
        break
    break

cv2.imshow('C1 Epoch1 (220x220)', x1[:,:,0]/np.max(x1[:,:,0]))
cv2.imshow('P1 Epoch1 (110x110)', y1[:,:,0]/np.max(y1[:,:,0]))
cv2.imshow('C2 Epoch1 (108x108)', x1[:,:,0]/np.max(x1[:,:,0]))
cv2.imshow('P2 Epoch1 (36x36)', y1[:,:,0]/np.max(y1[:,:,0]))

cv2.waitKey(0)
cv2.destroyAllWindows()
