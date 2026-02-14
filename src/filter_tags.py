import numpy as np
import cv2 

from convolution import convolve2d

def greyscale(image): #convert to grey scale
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    gray = 0.299*R + 0.587*G + 0.114*B
    return gray.astype(np.uint8)  #u means no negative int for integers only and 8 for 8-bit


def gaussian_smoothing(gray_img): #normal guassain blurring before finding edges
    kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ], dtype=np.float32) / 273

    smoothed_img = convolve2d(gray_img, kernel) #colvoluting kernel over image
    return smoothed_img