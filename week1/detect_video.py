
from ultralytics import YOLO

# Charge le modèle pré-entraîné
model = YOLO("yolov8n.pt")

# Détect on a local video inside your project file  (classes=[0] = personnes seulement) or you can use C:/Users/nom_user/Videos/myvideo.mp4
results = model("test_video.mp4", save=True, classes=[0])

print("Vidéo annotée sauvegardée dans runs/detect/")
