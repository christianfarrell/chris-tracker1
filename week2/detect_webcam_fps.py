

import cv2
import time
from ultralytics import YOLO

# load the model 
model = YOLO("yolov8n.pt")

# Open the webcam  (0 = integrated, 1 = extern)
cap = cv2.VideoCapture(1)

print("Webcam démarrée — appuie sur Q pour quitter")

# first variable to help calculate the fps 
prev_time = 0

while True:
    # read a webcam's frame 
    ret, frame = cap.read()
    if not ret:
        print("Erreur webcam")
        break

    # Calcul  FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # YOLO's detection on the frame 
    results = model(frame, verbose=False)

    # Draw the boxes' boundaries 
    annotated = results[0].plot()

    # show the fps on the window 
    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show the window 
    cv2.imshow("YOLOv8 - Detection temps réel", annotated)

    # Quit with q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
