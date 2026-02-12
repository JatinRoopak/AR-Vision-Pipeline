import cv2
import numpy as np

from filter_tags import greyscale, gaussian_smoothing
from sobel_edge import sobel_edge_detect
from detect_tags import get_tag_corners
from transformation import homography, warp_perspective_fast
from decode_tags import split_into_smaller_grid, find_tag_id


def main():
    camera = cv2.VideoCapture(0) #open webcam

    if not camera.isOpened():
        print('Camera cannot be opened')
        return
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        grayscaled_frame = greyscale(frame) #do greyscaling
        smoothed_frame = gaussian_smoothing(grayscaled_frame) #smoothout for better edge detection
        edges = sobel_edge_detect(smoothed_frame) #applying edge detection

        candidates = get_tag_corners(edges)

        debug_frame = frame.copy()

        active_tag_found = False

        if len(candidates) > 0:
            for i in range(min(len(candidates), 3)): #run for largest 3 tags found

                tag_corners = candidates[0]
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
                    cv2.imshow("4. Un-Warped Tag", flat_tag)

                    #finding the id
                    tag_grid = split_into_smaller_grid(flat_tag)
                    tag_id, n_rotations = find_tag_id(tag_grid)

                    if tag_id is not None:
                        fixed_corners = np.roll(tag_corners, -n_rotations, axis=0) #for the top left corner to always be at indec 0 rotate
                        
                        #info_grafics
                        cx, cy = int(np.mean(tag_corners[:, 0])), int(np.mean(tag_corners[:, 1]))
                        cv2.putText(debug_frame, f"ID: {tag_id}", (cx-20, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        first_pt = fixed_corners[0].astype(int)
                        cv2.circle(debug_frame, tuple(first_pt), 10, (255, 0, 0), -1)

                        cv2.imshow("4, unwarped tag", flat_tag)
                        pts = tag_corners.astype(int)
                        cv2.polylines(debug_frame, [pts.reshape(-1,1,2)], True, (0,0,255), 5) #paint main tag boxes red
                        break
                    else:
                        print("decoding is failing")
      
                
        for points in candidates:
            points = points.astype(int)
            cv2.polylines(debug_frame, [points.reshape(-1, 1, 2)], True, (0, 255, 0), 3) #draw green lines around box
            
            labels = ["0", "1", "2", "3"]
            for i, (x,y) in enumerate(points):
                cv2.circle(debug_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(debug_frame, labels[i], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("AR Tag Detection", debug_frame)
        cv2.imshow("Edge Map", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()