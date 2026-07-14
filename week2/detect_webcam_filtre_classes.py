
import cv2
import time
from ultralytics import YOLO

# Charge le modèle
model = YOLO("yolov8n.pt")

# Ouvre la webcam (0 = intégrée, 1 = externe)
cap = cv2.VideoCapture(1)

# Classes à détecter (0=personne, 39=bouteille, 67=téléphone)
CLASSES = [0, 39, 67]

print("Webcam démarrée — appuie sur Q pour quitter")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur webcam")
        break

    # Calcule le FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Détection sur les classes choisies seulement
    results = model(frame, verbose=False, classes=CLASSES)

    # Dessine les bounding boxes
    annotated = results[0].plot()

    # Affiche le FPS en vert
    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affiche la fenêtre
    cv2.imshow("YOLOv8 - Detection temps réel", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()