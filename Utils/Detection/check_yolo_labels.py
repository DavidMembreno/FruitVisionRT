from ultralytics import YOLO

model = YOLO("runs/detect/fruit_detector_v158/weights/best.pt")


for i, name in model.names.items():
    print(f"{i}: {name}")

print("\nTotal classes:", len(model.names))
