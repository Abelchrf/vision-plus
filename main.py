from ultralytics import YOLO
import cv2
import os

# Charger le modèle YOLOv8 (version nano pour la vitesse)
model = YOLO("yolov8n.pt")

# Image d’entrée
input_path = "data/input/test.jpg"
output_dir = "data/output/"
os.makedirs(output_dir, exist_ok=True)

# Prédiction
results = model(input_path, save=True, project=output_dir, name="result")

# Affichage
for result in results:
    boxes = result.boxes.xyxy
    print("Détections :", len(boxes), "objets trouvés.")

# Si tu veux afficher l’image directement :
image = cv2.imread(f"{output_dir}/result/test.jpg")
cv2.imshow("Résultat YOLOv8", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

