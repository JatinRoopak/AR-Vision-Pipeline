import cv2 
import numpy as np

def order_points(pts): #homography require matched pairs of corner points to map correctly
    rectangle = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis = 1)
    rectangle[0] = pts[np.argmin(s)]  #top left
    rectangle[2] = pts[np.argmax(s)]  #bottom right 
    diff = np.diff(pts, axis=1)
    rectangle[1] = pts[np.argmin(diff)] #top right 
    rectangle[3] = pts[np.argmax(diff)] #bottom left  

    return rectangle

def get_tag_corners(bianry_img):
    contours, _ = cv2.findContours(bianry_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue

        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)

            ordered_pts = order_points(pts)

            area = cv2.contourArea(cnt)
            candidates.append((area, ordered_pts))

    candidates.sort(key=lambda x: x[0], reverse=True) #only take the tag which is the largest
    return [c[1] for c in candidates]