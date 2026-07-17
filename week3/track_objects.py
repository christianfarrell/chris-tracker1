
import cv2
import time
from ultralytics import YOLO 

#load the model 
model = YOLO("yolov8n.pt")

#open the webcam , replace 1 by 0 if you use your computer integrated cam 
cap = cv2.VideoCapture(1)

#dictionary to stock up the objects trajectories 
trajectoires = {}

print("track started - press q to quit")

prev_time = 0 

while True:
    ret , frame = cap.read()
    if not ret:
        print("Webcam Error")
        break
    #calcul of FPS 
    curr_time =time.time()
    fps = 1/(curr_time - prev_time) 
    prev_time = curr_time
    
     # follpow with byteTrack , each object receive an unique id
    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          verbose=False, classes=[0])

    # draw the bounding boxes 
    annotated = results[0].plot()

    # draw the trajectory of eeach object 
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, obj_id in zip(boxes, ids):
            # box,s center , as the origin 
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]/2 ))

            # Add the trajectory'center as the box's center
            if obj_id not in trajectoires:
                trajectoires[obj_id] = []
            trajectoires[obj_id].append((cx, cy))

            # Limit the trajectory to 100 points
            if len(trajectoires[obj_id]) > 100:
                trajectoires[obj_id].pop(0)

            # Draw the trajectory in yellow
            for i in range(1, len(trajectoires[obj_id])):
                cv2.line(annotated, trajectoires[obj_id][i-1],
                         trajectoires[obj_id][i], (0, 255, 255), 2)

    # show the fps 
    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # show the window
    cv2.imshow("YOLOv8 + ByteTrack - Suivi temps réel", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()