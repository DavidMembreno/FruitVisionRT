import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

USE_PI_CAMERA = False

DETECTION_CLASSES = [
    'apple', 'banana', 'orange', 'strawberry', 'grapes', 'watermelon',
    'lemon', 'kiwi', 'mango', 'peach', 'pineapple', 'pomegranate',
    'tomato', 'avocado', 'cucumber', 'pear', 'cherry'
]

yolo_model = YOLO("runs/detect/fruit_detector_v158/weights/best.pt")

if USE_PI_CAMERA:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.start()
    def get_frame():
        return picam2.capture_array()
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    def get_frame():
        ret, frame = cap.read()
        return frame if ret else None

prev_time = time.time()

while True:
    frame = get_frame()
    if frame is None:
        continue

    results = yolo_model.predict(frame, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        det_label = DETECTION_CLASSES[cls_id]
        conf = float(box.conf[0])
        label = f"{det_label} ({conf:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_y = max(y1 + 20, 10)
        cv2.rectangle(frame, (x1, label_y - 20), (x1 + 160, label_y), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("FruitVisionRT - YOLO Only", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

if not USE_PI_CAMERA:
    cap.release()
cv2.destroyAllWindows()
