import cv2
import numpy as np

from filter_tags import greyscale, gaussian_smoothing
from sobel_edge import sobel_edge_detect
from detect_tags import get_tag_corners


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