
from ultralytics import YOLO

# Charge le modèle pré-entraîné
model = YOLO("yolov8n.pt")

# Détection sur une image en ligne
results = model("https://ultralytics.com/images/bus.jpg", save=True)

# Affiche les résultats dans le terminal
for box in results[0].boxes:
    print("Classe    :", results[0].names[int(box.cls)])
    print("Confiance :", round(float(box.conf), 2))
    print("Coords    :", box.xyxy)
    print("---")