import cv2 
from ultralytics import YOLO

# load the model 
model = YOLO("yolov8n.pt")

#open the webcam (0 is the default webcam , your computer camera)
cap = cv2.VideoCapture(1)

print("webcam started - press q to quti")

while True :
    ret , frame = cap.read()
    if not ret :
        print("webcam error")
        break
    # frame detection
    results = model(frame , verbose = False)

    #show the result

    annotated = results[0].plot()
    cv2.imshow("YOLOV8 - real time detection" , annotated)

    #quit with q

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break


cap.release()
cv2.destroyAllWindows() 
    