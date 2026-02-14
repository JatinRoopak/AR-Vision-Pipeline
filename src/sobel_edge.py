import cv2 
import numpy as np 

from convolution import convolve2d

#sobel matrices from lecture
def sobel_edge_detect(gray_img):
    sobelx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32) / 8.0
    
    sobely = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1,-2,-1]], dtype=np.float32) / 8.0
    
    gx = convolve2d(gray_img, sobelx) #convolve the sobel matrix on the image
    gy = convolve2d(gray_img, sobely) 

    gradient_magnitude = np.sqrt(gx**2 + gy**2) 

    #normalising to the range 0-255
    gradient_magnitude  = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    #if the edge strength is weak ignore it. It might be a noise. (<50)
    _, edge_map = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

    return edge_map

def compute_derivatives(gray_img): #finding Ix and Iy
    sobelx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32) / 8.0
    
    sobely = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1,-2,-1]], dtype=np.float32) / 8.0
    
    Ix = convolve2d(gray_img, sobelx)
    Iy = convolve2d(gray_img, sobely)
    
    return Ix, Iy

