import cv2 
import numpy as np

from transformation import homography, warp_perspective_fast
from sobel_edge import sobel_edge_detect
from filter_tags import greyscale, gaussian_smoothing
from decode_tags import split_into_smaller_grid, find_tag_id

def order_points(pts): #homography require matched pairs of corner points to map correctly
    rectangle = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis = 1)
    rectangle[0] = pts[np.argmin(s)]  #top left
    rectangle[2] = pts[np.argmax(s)]  #bottom right 
    diff = np.diff(pts, axis=1)
    rectangle[1] = pts[np.argmin(diff)] #top right 
    rectangle[3] = pts[np.argmax(diff)] #bottom left  

    return rectangle

def get_tag_corners(binary_img):
    contours, herarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    if herarchy is None:
        return []

    herarchy = herarchy[0] 
    candidates = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        if area < 1000: #ignore tiny noises
            continue
        if cv2.contourArea(cnt) < 1000:
            continue
        
        current_herarchy = herarchy[i]
        if current_herarchy[2] == -1: #if shape has no data inside it ignore it 
            continue

        #shape approximation filter
        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            ordered_pts = order_points(pts)

            width_A = np.linalg.norm(ordered_pts[0]-ordered_pts[1])
            height_A = np.linalg.norm(ordered_pts[0]-ordered_pts[3])

            if height_A == 0:
                continue
            aspect_ratio = width_A/height_A

            if 0.8 <= aspect_ratio <= 1.2: #check aspect ratio
                candidates.append((area, ordered_pts))

    candidates.sort(key=lambda x: x[0], reverse=True) #only take the tag which is the largest
    return [c[1] for c in candidates]

def detect_ar_tags(frame):
    detected_tags = []

    greyscaled_frame = greyscale(frame)
    gaussian_smoothed_frame = gaussian_smoothing(greyscaled_frame)
    edges = sobel_edge_detect(gaussian_smoothed_frame)
    candidates = get_tag_corners(edges)

    if len(candidates) > 0:
        for tag_corners in candidates[:10]: #run for largest 10 potrntial tags found
            # tag_corners = candidates[i]
            flat_corner = np.array([
                [0,0],
                [160, 0],
                [160, 160],
                [0, 160]
            ], dtype="float32")

            H = homography(tag_corners, flat_corner)

            if H is not None:
                #warping the color frame to see tag upright
                flat_tag = warp_perspective_fast(frame, H, (160, 160))

                #finding the id
                tag_grid = split_into_smaller_grid(flat_tag)
                tag_id, n_rotations = find_tag_id(tag_grid)

                if tag_id is not None:
                    fixed_corners = np.roll(tag_corners, -n_rotations, axis=0) #for the top left corner to always be at indec 0 rotate
                    
                    tag_info = {
                        "id": tag_id,
                        "corners": fixed_corners,   # The 4 corners in correct order
                        "center": (int(np.mean(tag_corners[:, 0])), int(np.mean(tag_corners[:, 1]))),
                        "homography": H,
                        "warped_image": flat_tag,
                        "orientation": n_rotations
                    }
                    detected_tags.append(tag_info)
    
    return detected_tags, edges