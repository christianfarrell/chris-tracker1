

import cv2
import time
from ultralytics import YOLO

# Charge le modèle
model = YOLO("yolov8n.pt")

# Ouvre la webcam (0 = intégrée, 1 = externe)
cap = cv2.VideoCapture(1)

print("Webcam démarrée — appuie sur Q pour quitter")

# Variables pour calculer le FPS
prev_time = 0

while True:
    # Lit une frame de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur webcam")
        break

    # Calcule le FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Détection YOLO sur la frame
    results = model(frame, verbose=False)

    # Dessine les bounding boxes
    annotated = results[0].plot()

    # Affiche le FPS sur la fenêtre
    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affiche la fenêtre
    cv2.imshow("YOLOv8 - Detection temps réel", annotated)

    # Quitter en appuyant sur Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libère les ressources
cap.release()
cv2.destroyAllWindows()