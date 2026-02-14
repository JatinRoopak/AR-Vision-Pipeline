import cv2
import numpy as np

from detect_tags import detect_ar_tags
from overlay_img import overlay_image

def main():
    camera = cv2.VideoCapture(0) #open webcam
    # camera = cv2.VideoCapture('multipleTags.mp4') #use pre shooted video
    template_img = cv2.imread("flipped_quadruped.png")

    if not camera.isOpened():
        print('Camera cannot be opened')
        return
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        tags, edge = detect_ar_tags(frame)
        debug_frame = frame.copy()

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

            debug_frame = overlay_image(debug_frame, template_img, corners)
            
            #Info Graphics
            cv2.polylines(debug_frame, [corners.reshape(-1,1,2)], True, (0, 0, 255), 3)

            cv2.putText(debug_frame, f"ID: {tag_id}", (cx-20, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.circle(debug_frame, tuple(corners[0]), 10, (255, 0, 0), -1)

            labels = ["0", "1", "2", "3"]
            for i, point in enumerate(corners):
                cv2.putText(debug_frame, labels[i], (point[0] - 10, point[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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