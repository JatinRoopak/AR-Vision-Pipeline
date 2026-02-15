import cv2 
import numpy as np 

from transformation import homography, projection_matrix
from obj_loader import render

# the k matrix we got for our camera
K = np.array([
    [518.9357033,   0.,         307.72634151],
    [  0.,         520.45365893, 212.95823523],
    [  0.,           0.,           1.        ]
])

#the k matrix for pre render video
# K = np.array([[1406.08415449821,    2.206797873085990,      1014.136434174160],
#                 [0,                   1417.99930662800,       566.347754321696],
#                 [0,                   0,                      1]])


def overlay_3d_object(background_frame, obj_model, tag_corners, tag_size = 200):
    src_points_3d = np.array([
        [0, 0],
        [tag_size, 0],
        [tag_size, tag_size],
        [0, tag_size]
    ], dtype="float32")

    dst_points = tag_corners.astype("float32")
    H_3d = homography(src_points_3d, dst_points)

    if H_3d is not None:
        P = projection_matrix(H_3d, K)

        #Create a dummy model to safely pass dimensions to the renderer
        dummy_model = np.zeros((tag_size, tag_size), dtype=np.uint8)
    
        background_frame = render(background_frame, obj_model, P, dummy_model, color=True)

    return background_frame