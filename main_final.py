import cv2
import torch
import numpy as np
import time
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

USE_PI_CAMERA = False
IMG_SIZE = 128

DETECTION_CLASSES = [
    'apple',        # 0
    'banana',       # 1
    'orange',       # 2
    'strawberry',   # 3
    'grapes',       # 4
    'watermelon',   # 5
    'lemon',        # 6
    'kiwi',         # 7
    'mango',        # 8
    'peach',        # 9
    'pineapple',    # 10
    'pomegranate',  # 11
    'tomato',       # 12
    'avocado',      # 13
    'cucumber',     # 14
    'pear',         # 15
    'cherry'        # 16
]


CLASSIFIABLE_CLASSES = [
    'apple', 'avocado', 'banana', 'bell pepper', 'carrot',
    'cucumber', 'grapes', 'kiwi', 'lemon', 'mango',
    'orange', 'pear', 'pineapple', 'pomegranate', 'strawberry',
    'tomato', 'watermelon'
]



# Load YOLO and TFLite models
yolo_model = YOLO("Models/detector/best.pt")
interpreter = tf.lite.Interpreter(model_path="Models/classifier/clv2_mobilenetv2_smooth05_v2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Camera input
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

# Classifier prediction
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

# Main App class
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("FruitVisionRT")
        self.root.configure(bg="white")

        self.video_frame = tk.Label(borderwidth=2, relief="groove")
        self.video_frame.pack(padx=10, pady=10)

        self.label_output = tk.Label(root, text="", font=("Times New Roman", 14), bg="white")
        self.label_output.pack(pady=(0, 10))

        self.mode = tk.StringVar(value="only_yolo")

        modes = [
            ("Only YOLO", "only_yolo"),
            ("Only Classifier", "only_cl"),
            ("Classifier Crop", "classifier_crop"),
            ("Mixed Decision", "mixed_decision")
        ]

        left_frame = tk.Frame(root, bg="white")
        left_frame.pack(side=tk.LEFT, anchor=tk.NW, padx=10, pady=10)

        for text, value in modes:
            ttk.Radiobutton(left_frame, text=text, variable=self.mode, value=value).pack(anchor=tk.W)

        self.update()

    def update(self):
        frame = get_frame()
        if frame is None:
            self.root.after(10, self.update)
            return

        start_time = time.time()
        display_text = ""
        mode = self.mode.get()

        if mode == "only_yolo":
            results = yolo_model.predict(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = DETECTION_CLASSES[cls_id]
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                display_text += f"YOLO: {label} ({conf:.2f})\n"

        elif mode == "only_cl":
            h, w, _ = frame.shape
            size = min(h, w) // 2
            x1 = w // 2 - size // 2
            y1 = h // 2 - size // 2
            x2 = x1 + size
            y2 = y1 + size
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] and crop.shape[1]:
                label, conf = classify_image(crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                display_text += f"Classifier: {label} ({conf:.2f})\n"

        elif mode == "classifier_crop":
            results = yolo_model.predict(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] and crop.shape[1]:
                    label, conf = classify_image(crop)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    display_text += f"Crop: {label} ({conf:.2f})\n"

        elif mode == "mixed_decision":
            results = yolo_model.predict(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_label = DETECTION_CLASSES[int(box.cls[0])]
                yolo_conf = float(box.conf[0])
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] and crop.shape[1]:
                    cl_label, cl_conf = classify_image(crop)
                    final_label, final_conf = (yolo_label, yolo_conf) if yolo_conf >= cl_conf else (cl_label, cl_conf)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"{final_label} ({final_conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    display_text += f"Mixed: {final_label} ({final_conf:.2f})\n"

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        self.label_output.config(text=display_text.strip())

        self.root.after(10, self.update)

    def on_close(self):
        if not USE_PI_CAMERA:
            cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
