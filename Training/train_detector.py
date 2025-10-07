import os

from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    # Root of the project (FruitVisionRT)
    ROOT_DIR = os.path.dirname(__file__)

    # Direct YAML path
    DATA_PATH = os.path.join(ROOT_DIR, "..", "fruit_data.yaml")  # Adjusted path

    model.train(
        data=DATA_PATH,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=7,
        device=0,  # 0 = GPU
        name="fruit_detector_v15"
    )


if __name__ == "__main__":
    main()
