import cv2
import numpy as np
def relu (image, max=None, threshold = 0): #image in the form of numpy array
    image[image < threshold] = threshold
    if max is not None:
        image[image > max] = max
    return image


