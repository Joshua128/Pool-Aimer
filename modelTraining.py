from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os



def circleCoords(img, model, conf_thresh=0.45):
    
    results = model(img)
    boxes = results[0].boxes

    # Get coordinates and confidence scores
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    xyxy_round = np.round(xyxy).astype(int)

    # Filter out boxes with low confidence
    kept_boxes = []
    for (x1, y1, x2, y2), conf in zip(xyxy_round, confs):
        if conf >= conf_thresh:
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            kept_boxes.append([x1, y1, x2, y2])

    return np.array(kept_boxes)






"""
vid = cv2.VideoCapture(0)
while True:
    ret, frame = vid.read()
    if not ret:
        break

    # run YOLO on the raw BGR numpy array
    results = model(frame)[0]

    # draw the detections
    output = results.plot()

    # display
    cv2.imshow("OUTPUT", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
vid.release()
cv2.destroyAllWindows()

cv2.imwrite("pred_one.jpg", annotated)
print("Saved annotated image to pred_one.jpg")

"""