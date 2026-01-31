import cv2
import numpy as np
from sobel_edge import compute_derivatives
from filter_tags import guassian_smoothing

def compute_harris_responses(gray_img):
    Ix, Iy = compute_derivatives(gray_img)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    Sxx = guassian_smoothing(Ixx)
    Syy = guassian_smoothing(Iyy)
    Sxy = guassian_smoothing(Ixy)

    det_M = (Sxx * Syy) - (Sxy**2)
    trace_M = Sxx + Syy
    k = 0.04
    R = det_M - k*(trace_M**2)

    return R

