import os
from ultralytics import YOLO

def convert_yolo_to_tflite(weight_path):
    model = YOLO(weight_path)
    model.export(format='tflite')

if __name__ == "__main__":
    pt_path = r"C:\Users\DM77\Documents\FruitVisionRT\runs\detect\fruit_detector_v158\weights\best.pt"
    convert_yolo_to_tflite(pt_path)
#C:\Users\DM77\Documents\FruitVisionRT\models\tflite_yolo\model_float32.tflite
