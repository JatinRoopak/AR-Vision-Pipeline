import cv2 
import numpy as np

from transformation import homography, warp_perspective_fast

def overlay_image(background_frame, imaage, tag_corners):
    h_img, w_img = imaage.shape[:2]

    #corners of image in order
    src_points = np.array([
        [0,0],
        [w_img, 0],
        [w_img, h_img],
        [0, h_img]
    ], dtype="float32")

    destination_points = tag_corners.astype("float32")

    H = homography(src_points, destination_points)

    if H is not None:
        #inversing warp
        h_bg , w_bg = background_frame.shape[:2]
        warped_overlay = warp_perspective_fast(imaage, H, (w_bg, h_bg))

        #clear space for the image overlaying by creating a black spot 
        mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
        cv2.fillConvexPoly(mask, destination_points.astype(int), 255)
        background_frame = cv2.bitwise_and(background_frame, background_frame, mask=cv2.bitwise_not(mask))
        
        background_frame = cv2.add(background_frame, warped_overlay)
    
    return background_frame