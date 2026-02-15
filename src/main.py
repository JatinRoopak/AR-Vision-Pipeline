import cv2
import numpy as np

from detect_tags import detect_ar_tags
from overlay_img import overlay_image
from transformation import homography, projection_matrix
from obj_loader import OBJ, render
from filter_tags import greyscale
from overlay_model import overlay_3d_object

def main():
    camera = cv2.VideoCapture(0) #open webcam
    # camera = cv2.VideoCapture('multipleTags.mp4') #use pre shooted video
    template_img = cv2.imread("flipped_quadruped.png")
    template_gray = greyscale(template_img)

    wolf = OBJ('model1.obj', swapyz=True)
    scale_factor = .10 # You may need to tune this up or down so the wolf fits perfectly!
    wolf.vertices = [[v[0]*scale_factor, v[1]*scale_factor, v[2]*scale_factor] for v in wolf.vertices]

    if not camera.isOpened():
        print('Camera cannot be opened')
        return
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        tags, edge, debug_frame = detect_ar_tags(frame)

        seen_centers = []
        current_warped_tag = None

        for tag in tags:
            tag_id = tag["id"]
            corners = tag["corners"].astype(int)
            cx, cy = tag["center"]

            #checking mechnism for the tags if 2 tags have same id they still be detected deferently
            is_duplicate = False
            for(existing_cx, existing_cy) in seen_centers:
                distance = np.sqrt((cx - existing_cx)**2 + (cy - existing_cy)**2)
                if distance<20: 
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            seen_centers.append((cx, cy))

            if "warped_image" in tag:
                current_warped_tag = tag["warped_image"]

            # debug_frame = overlay_image(debug_frame, template_img, corners)
            debug_frame = overlay_3d_object(debug_frame, wolf, corners, tag_size=350)

            h_img, w_img = template_gray.shape[:2]
            tag_size = 350

        # cv2.imshow("Edge Map", edge)
        cv2.imshow("AR Tag Detection", debug_frame)


        if current_warped_tag is not None:
            cv2.imshow("Warped Tag", current_warped_tag)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()