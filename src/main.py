import cv2
import numpy as np

from detect_tags import detect_ar_tags
from overlay_img import overlay_image
from obj_loader import OBJ
from filter_tags import greyscale
from overlay_model import overlay_3d_object

def main():
    camera = cv2.VideoCapture(0) #open webcam
    # camera = cv2.VideoCapture('multipleTags.mp4') #use pre shooted video
    template_img = cv2.imread("flipped_quadruped.png")
    template_gray = greyscale(template_img)

    wolf = OBJ('model3.obj', swapyz=True)
    scale_factor = .10 # You may need to tune this up or down so the wolf fits perfectly!
    wolf.vertices = [[v[0]*scale_factor, v[1]*scale_factor, v[2]*scale_factor] for v in wolf.vertices]

    if not camera.isOpened():
        print('Camera cannot be opened')
        return
    
    corner_history = [] #we will store our frames to furthur apply smoothing to it
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        tags, edge, debug_frame, corner_history = detect_ar_tags(frame, corner_history)

        current_warped_tag = None

        for tag in tags:
            corners = tag["corners"].astype(int)

            if "warped_image" in tag:
                current_warped_tag = tag["warped_image"]

            # debug_frame = overlay_image(debug_frame, template_img, corners)
            debug_frame = overlay_3d_object(debug_frame, wolf, corners, tag_size=350)

        # cv2.imshow("Edge Map", edge)
        cv2.imshow("AR Tag Detection", debug_frame)

        # if current_warped_tag is not None:
        #     cv2.imshow("Warped Tag", current_warped_tag)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()