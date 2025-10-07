import cv2
import numpy as np
import tensorflow as tf
import time

USE_PI_CAMERA = False
IMG_SIZE = 128


CLASSIFIABLE_CLASSES = [
    'apple', 'avocado', 'banana', 'bell pepper', 'carrot', 'cucumber',
    'grapes', 'kiwi', 'lemon', 'mango', 'orange', 'pear', 'pineapple',
    'pomegranate', 'strawberry', 'tomato', 'watermelon'
]


interpreter = tf.lite.Interpreter(model_path="classification/clv2_mobilenetv2_smooth05_v2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#camera source
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

def classify_image(image):
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    class_id = int(np.argmax(output))
    confidence = float(output[0][class_id])
    return CLASSIFIABLE_CLASSES[class_id], confidence

# Main loop
prev_time = time.time()
while True:
    frame = get_frame()
    if frame is None:
        continue

    h, w, _ = frame.shape
    size = min(h, w) // 2
    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2
    x2 = x1 + size
    y2 = y1 + size

    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        continue

    class_label, class_conf = classify_image(crop)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    label = f"{class_label} ({class_conf:.2f})"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Fruit Classifier Only", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

if not USE_PI_CAMERA:
    cap.release()
cv2.destroyAllWindows()
