import cv2 
import numpy as np 

#sobel matrices from lecture
def sobel_edge_detect(gray_img):
    sobelx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32) / 8.0
    
    sobely = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1,-2,-1]], dtype=np.float32) / 8.0
    
    gx = cv2.filter2D(gray_img, cv2.CV_32F, sobelx) #convolve the sobel matrix on the image
    gy = cv2.filter2D(gray_img, cv2.CV_32F, sobely) 

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
    
    Ix = cv2.filter2D(gray_img, cv2.CV_32F, sobelx)
    Iy = cv2.filter2D(gray_img, cv2.CV_32F, sobely)
    
    return Ix, Iy

