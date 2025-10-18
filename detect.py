from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

input_folder = "data/input/"
output_folder = "data/output/batch_results/"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(('.jpg', '.png')):
        path = os.path.join(input_folder, file)
        print(f"Traitement de {file} ...")
        model(path, save=True, project=output_folder, name="visionplus")

