
from ultralytics import YOLO

# load the model
model = YOLO("yolov8n.pt")

# Detect on a picture  , if it is a local picture use C:/Users/nom_user/pictures/photo.jpg
results = model("https://ultralytics.com/images/bus.jpg", save=True)

# show the results on the terminal 
for box in results[0].boxes:
    print("Classe    :", results[0].names[int(box.cls)])
    print("Confiance :", round(float(box.conf), 2))
    print("Coords    :", box.xyxy)
    print("---")
