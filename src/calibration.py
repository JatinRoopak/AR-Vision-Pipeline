import cv2
import numpy as np
import glob

from filter_tags import greyscale

#to know the resolution of camera
# cap = cv2.VideoCapture(0)
# print(cap.get(3), cap.get(4))  # 3=Width, 4=Height
# cap.release()


checkerboard = (7, 10)
sqaure_size = 15 #should in mm

object_points = [] #3d points in world 
image_points = [] #2d points in image plane

objp = np.zeros((checkerboard[0]*checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2) #without T our grid will be column major open cv expect row major
objp = objp*sqaure_size #real world scaling 

cap = cv2.VideoCapture(0)
#enter resolution of your camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("press c to capture")
print("press q to calibrate")

frame_captured = 0

while True:
    ret, frame = cap.read() #capture frame
    if not ret: break

    gray = greyscale(frame)

    found, corners = cv2.findChessboardCorners(gray, checkerboard, None)

    debug_frame = frame.copy()

    if found:
        #subpixel is for correcting the problem of falling corner at a 10.5 pixel (does not exist )
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(debug_frame, checkerboard, corners2, found)
    cv2.putText(debug_frame, f"Captured: {frame_captured}", (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("calibration", debug_frame)
    key = cv2.waitKey(1)

    if key == ord('c') and found:
        object_points.append(objp)
        image_points.append(corners2)
        frame_captured += 1
        print(f"Frame {frame_captured} captured")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if frame_captured > 5:
    print("Calibrating...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    print("K = np.array(")
    print(np.array_repr(K))
    print(")")
    #also giving the distortion matrix the object way drift if comes  in corner of frame. 
    print("dist = np.array(")
    print(np.array_repr(dist))
    print(")")
else:
    print("Not enough frames.")