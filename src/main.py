import numpy as np

def greyscale(image): #convert to grey scale
    B = image[:,:,0]
    G = image[:,:,1]
    R = image[:,:,2]

    gray = 0.299*R + 0.587*G + 0.114*B
    return gray.astype(np.uint8)  #u means no negative int for integers only and 8 for 8-bit

def binarize(gray_image, threshold=127): #a threshold to differentiate black and white
    binary = np.zeros_like(gray_image)
    binary[gray_image>threshold] = 255 #more intensity than the 127 turn white
    return binary