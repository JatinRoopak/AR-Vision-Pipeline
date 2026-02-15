import cv2 
import numpy as np

from transformation import homography, warp_perspective_fast
from sobel_edge import sobel_edge_detect
from filter_tags import greyscale, gaussian_smoothing
from decode_tags import split_into_smaller_grid, find_tag_id

def order_points(pts): #homography require matched pairs of corner points to map correctly
    # rectangle = np.zeros((4, 2), dtype="float32")
    # s = pts.sum(axis = 1)
    # rectangle[0] = pts[np.argmin(s)]  #top left
    # rectangle[2] = pts[np.argmax(s)]  #bottom right 
    # diff = np.diff(pts, axis=1)
    # rectangle[1] = pts[np.argmin(diff)] #top right 
    # rectangle[3] = pts[np.argmax(diff)] #bottom left  

    pts = pts.reshape(4, 2)
    center = np.mean(pts, axis=0) #finad exact center of 4 poiints

    angles = np.arctan2(pts[:, 1]-center[1], pts[:, 0] - center[0])

    sorted_indices = np.argsort(angles) #sorting the points based on their angles


    # return rectangle
    return pts[sorted_indices].astype("float32")

def ema_smoothing(current_corners, current_center, previous_history, alpha = 0.4):
    best_diatnace = float('inf')
    best_previous_corners = None
    
    cx, cy = current_center

    for prev_tag in previous_history:
        px, py = prev_tag["center"]
        dist = np.sqrt(((cx - px)**2 + (cy - py)**2))

        if dist < 60 and dist < best_diatnace:
            best_diatnace = dist
            best_previous_corners = prev_tag["corners"]

    if best_previous_corners is not None:
        smoothed = (alpha * current_corners) + ((1.0 - alpha) * best_previous_corners)
        return smoothed
    else:
        #no tag was nearby in the last frame, treat as new
        return current_corners


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
        
        # current_herarchy = herarchy[i]
        # if current_herarchy[2] == -1: #if shape has no data inside it ignore it 
        #     continue

        #shape approximation filter
        epsilon = 0.05*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            ordered_pts = order_points(pts)

            width_A = np.linalg.norm(ordered_pts[0]-ordered_pts[1])
            height_A = np.linalg.norm(ordered_pts[0]-ordered_pts[3])

            if height_A == 0:
                continue
            aspect_ratio = width_A/height_A

            if 0.3 <= aspect_ratio <= 3: #check aspect ratio
                candidates.append((area, ordered_pts))

    candidates.sort(key=lambda x: x[0], reverse=True) #only take the tag which is the largest
    return [c[1] for c in candidates]

def detect_ar_tags(frame, previous_history):
    detected_tags = []
    debug_frame = frame.copy()
    greyscaled_frame = greyscale(frame)
    gaussian_smoothed_frame = gaussian_smoothing(greyscaled_frame)
    edges = sobel_edge_detect(gaussian_smoothed_frame)
    candidates = get_tag_corners(edges)

    new_history = []
    seen_centers = []

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
                    
                    cx = int(np.mean(fixed_corners[:, 0]))
                    cy = int(np.mean(fixed_corners[:, 1]))

                    is_duplicate = False
                    for (existing_cx, existing_cy) in seen_centers:
                        distance = np.sqrt((cx - existing_cx)**2 + (cy - existing_cy)**2)
                        if distance < 20: 
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue

                    #apply ema smoothing
                    fixed_corners = ema_smoothing(fixed_corners, (cx, cy), previous_history, alpha=0.4)

                    new_history.append({"center": (cx, cy), "corners": fixed_corners})

                    tag_info = {
                        "id": tag_id,
                        "corners": fixed_corners,   # The 4 corners in correct order
                        "center": (int(np.mean(tag_corners[:, 0])), int(np.mean(tag_corners[:, 1]))),
                        "warped_image": flat_tag,
                        "orientation": n_rotations
                    }
                    detected_tags.append(tag_info)

                    cx = int(np.mean(fixed_corners[:, 0]))
                    cy = int(np.mean(fixed_corners[:, 1]))

                    #Info graphics
                    corners_int = fixed_corners.astype(int)
                    
                    cv2.polylines(debug_frame, [corners_int.reshape(-1,1,2)], True, (0, 0, 255), 3)

                    cv2.putText(debug_frame, f"ID: {tag_id}", (cx-20, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    cv2.circle(debug_frame, tuple(corners_int[0]), 10, (255, 0, 0), -1)

                    labels = ["0", "1", "2", "3"]
                    for i, point in enumerate(corners_int):
                        cv2.putText(debug_frame, labels[i], (point[0] - 10, point[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            
    
    return detected_tags, edges, debug_frame, new_history
