import cv2
import torch
import numpy as np
import tensorflow as tf
import time
from ultralytics import YOLO

# Toggle controls
mixed_decision = False
classifier_crop = False
only_yolo = True
only_cl = False

USE_PI_CAMERA = False
IMG_SIZE = 128

det_classes = [
    'apple', 'avocado', 'banana', 'bell pepper', 'carrot', 'cherry', 'cucumber',
    'grapes', 'kiwi', 'lemon', 'mango', 'orange', 'peach', 'pear', 'pineapple',
    'pomegranate', 'strawberry', 'tomato', 'watermelon'
]

cl_classes = [
    'apple', 'avocado', 'banana', 'bell pepper', 'carrot', 'cucumber', 'grapes',
    'kiwi', 'lemon', 'mango', 'orange', 'pear', 'pineapple', 'pomegranate',
    'strawberry', 'tomato', 'watermelon'
]

cl_class_set = set(cl_classes)

yolo_model = YOLO("runs/detect/fruit_detector_v158/weights/best.pt")

interpreter = tf.lite.Interpreter(model_path="classification/clv2_mobilenetv2_smooth05_v2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    class_id = int(np.argmax(output))
    confidence = float(output[0][class_id])
    return cl_classes[class_id], confidence

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

    h, w, _ = frame.shape
    display_frame = frame.copy()

    if only_cl:
        size = min(h, w) // 2
        x1 = w // 2 - size // 2
        y1 = h // 2 - size // 2
        x2 = x1 + size
        y2 = y1 + size
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        label, conf = classify_image(crop)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display_frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    elif only_yolo:
        results = yolo_model.predict(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = det_classes[cls_id]
            conf = float(box.conf[0])
            if label not in cl_class_set:
                continue
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    elif classifier_crop:
        results = yolo_model.predict(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            label, conf = classify_image(crop)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display_frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    elif mixed_decision:
        results = yolo_model.predict(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_id = int(box.cls[0])
            det_label = det_classes[det_id]
            det_conf = float(box.conf[0])
            if det_label not in cl_class_set:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            cl_label, cl_conf = classify_image(crop)
            if cl_conf > det_conf:
                label, conf = cl_label, cl_conf
                color = (255, 255, 0)
            else:
                label, conf = det_label, det_conf
                color = (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("FruitVisionRT", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

if not USE_PI_CAMERA:
    cap.release()
cv2.destroyAllWindows()
