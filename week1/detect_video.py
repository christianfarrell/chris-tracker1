
from ultralytics import YOLO

# Charge le modèle pré-entraîné
model = YOLO("yolov8n.pt")

# Détection sur une vidéo locale (classes=[0] = personnes seulement)
results = model("test_video.mp4", save=True, classes=[0])

print("Vidéo annotée sauvegardée dans runs/detect/")