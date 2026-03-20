import cv2
import torch
import numpy as np
import time
import tensorflow as tf
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import deque
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

USE_PI_CAMERA = False
CLASSIFIER_INPUT_SIZE = 128

YOLO_CLASSES = [
    'apple', 'banana', 'orange', 'strawberry', 'grapes',
    'watermelon', 'lemon', 'kiwi', 'mango', 'peach',
    'pineapple', 'pomegranate', 'tomato', 'avocado',
    'cucumber', 'pear', 'cherry'
]

CLASSIFIER_CLASSES = [
    'apple', 'avocado', 'banana', 'bell pepper', 'carrot',
    'cucumber', 'grapes', 'kiwi', 'lemon', 'mango',
    'orange', 'pear', 'pineapple', 'pomegranate', 'strawberry',
    'tomato', 'watermelon'
]

# Colors
BG_MAIN    = "#0d0f14"
BG_SURFACE = "#13161e"
BG_CARD    = "#1a1e2a"
GREEN      = "#00ff88"
GREEN_DIM  = "#00aa55"
AMBER      = "#ffaa00"
TEXT       = "#e8eaf0"
TEXT_MUTED = "#6b7280"
TEXT_GREEN = "#a0f0c0"
BORDER     = "#252a38"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Load models at startup
yolo_model = YOLO("Models/detector/best.pt")
interpreter = tf.lite.Interpreter(model_path="Models/classifier/clv2_mobilenetv2_smooth05_v2.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Camera setup - swap to PiCamera if deploying on Pi
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
    resized = cv2.resize(image, (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(normalized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    class_id = int(np.argmax(output))
    confidence = float(output[0][class_id])
    return CLASSIFIER_CLASSES[class_id], confidence


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FruitVisionRT")
        self.configure(fg_color=BG_MAIN)
        self.geometry("1280x780")
        self.minsize(1100, 680)

        self.mode = ctk.StringVar(value="only_yolo")
        self.fps_history = deque(maxlen=60)
        self.frame_count = 0
        self.mode_buttons = {}

        self._build_layout()
        self.update_frame()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=320)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_feed()
        self._build_sidebar()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=0,
                              height=52, border_width=1, border_color=BORDER)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        title_area = ctk.CTkFrame(header, fg_color="transparent")
        title_area.grid(row=0, column=0, padx=20, pady=8, sticky="w")
        ctk.CTkLabel(title_area, text="●", font=("Courier New", 14), text_color=GREEN).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(title_area, text="FRUITVISION", font=("Courier New", 16, "bold"), text_color=TEXT).pack(side="left")
        ctk.CTkLabel(title_area, text="RT", font=("Courier New", 16, "bold"), text_color=GREEN).pack(side="left")

        status_area = ctk.CTkFrame(header, fg_color="transparent")
        status_area.grid(row=0, column=2, padx=20, pady=8, sticky="e")
        ctk.CTkLabel(status_area, text="CPU INFERENCE  |", font=("Courier New", 11), text_color=TEXT_MUTED).pack(side="right")
        ctk.CTkLabel(status_area, text="● LIVE", font=("Courier New", 11), text_color=GREEN).pack(side="right", padx=(12, 0))

    def _build_feed(self):
        feed = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=0,
                            border_width=1, border_color=BORDER)
        feed.grid(row=1, column=0, sticky="nsew", padx=(0, 1))
        feed.grid_rowconfigure(1, weight=1)
        feed.grid_columnconfigure(0, weight=1)

        top_bar = ctk.CTkFrame(feed, fg_color="transparent", height=32)
        top_bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(10, 0))
        ctk.CTkLabel(top_bar, text="CAMERA FEED", font=("Courier New", 10), text_color=TEXT_MUTED).pack(side="left")
        self.fps_label = ctk.CTkLabel(top_bar, text="FPS: --", font=("Courier New", 10, "bold"), text_color=GREEN)
        self.fps_label.pack(side="right")

        self.video_label = ctk.CTkLabel(feed, text="", fg_color=BG_MAIN, corner_radius=4)
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 12))

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=0,
                               border_width=1, border_color=BORDER)
        sidebar.grid(row=1, column=1, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(3, weight=1)

        self._build_mode_selector(sidebar)
        self._build_detection_panel(sidebar)
        self._build_fps_chart(sidebar)
        self._build_stats_panel(sidebar)

        ctk.CTkButton(
            sidebar, text="VIEW TRAINING RESULTS",
            font=("Courier New", 11, "bold"),
            height=36, corner_radius=4,
            fg_color=BG_CARD, hover_color=GREEN_DIM,
            text_color=GREEN, border_width=1, border_color=GREEN_DIM,
            command=self._open_results_viewer
        ).grid(row=4, column=0, sticky="ew", padx=12, pady=(4, 12))

    def _build_mode_selector(self, parent):
        card = self._make_card(parent, "DETECTION MODE")
        card.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        btn_grid = ctk.CTkFrame(card, fg_color="transparent")
        btn_grid.pack(fill="x", pady=(4, 0))
        btn_grid.grid_columnconfigure((0, 1), weight=1)

        modes = [
            ("YOLO Only",       "only_yolo"),
            ("Classifier Only", "only_cl"),
            ("Classifier Crop", "classifier_crop"),
            ("Mixed Decision",  "mixed_decision"),
        ]

        for i, (label, value) in enumerate(modes):
            is_active = self.mode.get() == value
            btn = ctk.CTkButton(
                btn_grid, text=label,
                font=("Courier New", 11), height=32, corner_radius=4,
                fg_color=GREEN_DIM if is_active else BG_MAIN,
                hover_color=GREEN_DIM,
                text_color=BG_MAIN if is_active else TEXT_MUTED,
                border_width=1, border_color=BORDER,
                command=lambda v=value: self._set_mode(v)
            )
            btn.grid(row=i // 2, column=i % 2, padx=3, pady=3, sticky="ew")
            self.mode_buttons[value] = btn

    def _build_detection_panel(self, parent):
        card = self._make_card(parent, "DETECTIONS")
        card.grid(row=1, column=0, sticky="ew", padx=12, pady=6)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x")

        self.det_rows = []
        for _ in range(5):
            row = ctk.CTkFrame(inner, fg_color="transparent", height=20)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)

            name = ctk.CTkLabel(row, text="", font=("Courier New", 12, "bold"),
                                text_color=GREEN, width=150, anchor="w")
            name.pack(side="left")

            conf = ctk.CTkLabel(row, text="", font=("Courier New", 11),
                                text_color=TEXT_MUTED, width=45, anchor="e")
            conf.pack(side="right")

            bar = ctk.CTkProgressBar(row, height=4, corner_radius=2,
                                     fg_color=BG_MAIN, progress_color=GREEN)
            bar.set(0)
            bar.pack(fill="x", pady=(2, 0))
            self.det_rows.append((name, conf, bar))

    def _build_fps_chart(self, parent):
        card = self._make_card(parent, "FPS HISTORY")
        card.grid(row=2, column=0, sticky="ew", padx=12, pady=6)

        self.chart_fig = Figure(figsize=(2.8, 1.2), dpi=90, facecolor=BG_CARD)
        self.chart_ax  = self.chart_fig.add_subplot(111)
        self._style_chart()

        canvas = FigureCanvasTkAgg(self.chart_fig, master=card)
        canvas.get_tk_widget().configure(bg=BG_CARD, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="x", pady=(4, 0))
        self.chart_canvas = canvas

    def _build_stats_panel(self, parent):
        card = self._make_card(parent, "SESSION STATS")
        card.grid(row=3, column=0, sticky="sew", padx=12, pady=(6, 6))

        self.stat_labels = {}
        rows = [
            ("Frames Processed", "frames"),
            ("Avg FPS",          "avg_fps"),
            ("Peak FPS",         "peak_fps"),
            ("Active Mode",      "mode_name"),
        ]
        for label, key in rows:
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label, font=("Courier New", 10),
                         text_color=TEXT_MUTED, anchor="w").pack(side="left")
            val = ctk.CTkLabel(row, text="--", font=("Courier New", 10, "bold"),
                               text_color=TEXT_GREEN, anchor="e")
            val.pack(side="right")
            self.stat_labels[key] = val

    def _make_card(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=6,
                             border_width=1, border_color=BORDER)
        ctk.CTkLabel(frame, text=title, font=("Courier New", 9),
                     text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(8, 4))
        ctk.CTkFrame(frame, fg_color=BORDER, height=1).pack(fill="x", padx=10, pady=(0, 8))
        return frame

    def _set_mode(self, value):
        self.mode.set(value)
        for v, btn in self.mode_buttons.items():
            if v == value:
                btn.configure(fg_color=GREEN_DIM, text_color=BG_MAIN)
            else:
                btn.configure(fg_color=BG_MAIN, text_color=TEXT_MUTED)

    def _style_chart(self):
        self.chart_ax.clear()
        self.chart_ax.set_facecolor(BG_CARD)
        self.chart_ax.tick_params(colors=TEXT_MUTED, labelsize=7)
        self.chart_ax.spines[:].set_color(BORDER)
        self.chart_fig.tight_layout(pad=0.4)

    def _redraw_chart(self):
        self._style_chart()
        if len(self.fps_history) > 1:
            xs = list(range(len(self.fps_history)))
            ys = list(self.fps_history)
            self.chart_ax.plot(xs, ys, color=GREEN, linewidth=1.2)
            self.chart_ax.fill_between(xs, ys, alpha=0.15, color=GREEN)
            self.chart_ax.set_ylim(0, max(ys) * 1.3)
        self.chart_canvas.draw()

    def _update_detections(self, detections):
        for i, (name_lbl, conf_lbl, bar) in enumerate(self.det_rows):
            if i < len(detections):
                label, conf = detections[i]
                name_lbl.configure(text=label.upper())
                conf_lbl.configure(text=f"{conf:.0%}")
                bar.set(conf)
                bar.configure(progress_color=GREEN if conf > 0.6 else AMBER)
            else:
                name_lbl.configure(text="")
                conf_lbl.configure(text="")
                bar.set(0)

    def _update_stats(self, fps):
        self.frame_count += 1
        self.stat_labels["frames"].configure(text=str(self.frame_count))
        if self.fps_history:
            self.stat_labels["avg_fps"].configure(text=f"{np.mean(self.fps_history):.1f}")
            self.stat_labels["peak_fps"].configure(text=f"{max(self.fps_history):.1f}")
        mode_display = {
            "only_yolo": "YOLO Only", "only_cl": "Classifier",
            "classifier_crop": "Crop", "mixed_decision": "Mixed"
        }
        self.stat_labels["mode_name"].configure(text=mode_display.get(self.mode.get(), "--"))

    def _open_results_viewer(self):
        if hasattr(self, "results_window") and self.results_window.winfo_exists():
            self.results_window.focus()
            return
        self.results_window = ResultsViewer(self)
        self.results_window.focus()

    def update_frame(self):
        frame = get_frame()
        if frame is None:
            self.after(10, self.update_frame)
            return

        start = time.time()
        detections = []
        mode = self.mode.get()

        if mode == "only_yolo":
            results = yolo_model.predict(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = YOLO_CLASSES[int(box.cls[0])]
                conf  = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 136), 2)
                cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 136), 2)
                detections.append((label, conf))

        elif mode == "only_cl":
            h, w, _ = frame.shape
            size = min(h, w) // 2
            x1 = w // 2 - size // 2
            y1 = h // 2 - size // 2
            x2, y2 = x1 + size, y1 + size
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] and crop.shape[1]:
                label, conf = classify_image(crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 0), 2)
                cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 170, 0), 2)
                detections.append((label, conf))

        elif mode == "classifier_crop":
            results = yolo_model.predict(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] and crop.shape[1]:
                    label, conf = classify_image(crop)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 170, 0), 2)
                    detections.append((label, conf))

        elif mode == "mixed_decision":
            results = yolo_model.predict(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_label = YOLO_CLASSES[int(box.cls[0])]
                yolo_conf  = float(box.conf[0])
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] and crop.shape[1]:
                    cl_label, cl_conf = classify_image(crop)
                    # Pick whichever model is more confident
                    if yolo_conf >= cl_conf:
                        final_label, final_conf = yolo_label, yolo_conf
                    else:
                        final_label, final_conf = cl_label, cl_conf
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    cv2.putText(frame, f"{final_label} {final_conf:.0%}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                    detections.append((final_label, final_conf))

        fps = 1.0 / (time.time() - start)
        self.fps_history.append(fps)
        self.fps_label.configure(text=f"FPS: {fps:.1f}")

        w = self.video_label.winfo_width()
        h = self.video_label.winfo_height()
        if w > 10 and h > 10:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((w, h), Image.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label._image = ctk_img

        self._update_detections(detections)
        self._update_stats(fps)

        if self.frame_count % 10 == 0:
            self._redraw_chart()

        self.after(10, self.update_frame)

    def on_close(self):
        if not USE_PI_CAMERA:
            cap.release()
        self.destroy()


class ResultsViewer(ctk.CTkToplevel):

    SECTIONS = [
        # Classifier plots
        {
            "title":  "INITIAL TRAINING CURVES — MobileNetV2 Classifier",
            "model":  "Model: MobileNetV2 (TFLite) · Task: 17-class fruit classification",
            "desc":   "Loss and accuracy curves from Phase 1 — frozen MobileNetV2 base layers, only the "
                      "classification head trained. Train and validation curves converge closely around "
                      "86-87%, confirming the pretrained ImageNet features transfer well to fruit recognition "
                      "without overfitting.",
            "metric": "Phase 1 result: ~87% val accuracy · Stable convergence · No significant overfitting.",
            "path":   "Training/Classifier Performance Data/training_plots(Initial_Training).png",
        },
        {
            "title":  "FULL TRAINING & VALIDATION CURVES — MobileNetV2 Classifier",
            "model":  "Model: MobileNetV2 (TFLite) · Task: 17-class fruit classification",
            "desc":   "Combined accuracy and loss curves across the full 60-epoch training run. "
                      "Both curves rise and converge steadily with minimal gap between train and validation, "
                      "confirming the model generalizes well to unseen data. "
                      "Early stopping was applied at peak validation performance.",
            "metric": "Final Val Accuracy: ~96% · Train/Val gap minimal · Trained on NVIDIA RTX 5070 (Blackwell).",
            "path":   "Training/Classifier Performance Data/TrainValidPlots.png",
        },
        {
            "title":  "CONFUSION MATRIX — MobileNetV2 Classifier",
            "model":  "Model: MobileNetV2 (TFLite) · Task: 17-class fruit classification",
            "desc":   "Raw prediction counts per class across the test set. Each row is a true label, "
                      "each column a predicted label. Strong diagonal values with minimal off-diagonal "
                      "entries confirm the classifier correctly identifies the vast majority of fruit samples. "
                      "Most confusion occurs between visually similar classes such as lemon and orange.",
            "metric": "Precision: 95.99% · Recall: 95.61% · F1: 95.79%",
            "path":   "Training/Classifier Performance Data/confusion_matrix.png",
        },
        {
            "title":  "PRECISION · RECALL · F1 PER CLASS — MobileNetV2 Classifier",
            "model":  "Model: MobileNetV2 (TFLite) · Task: 17-class fruit classification",
            "desc":   "Per-class breakdown of Precision, Recall, and F1 score. "
                      "Precision = of all predicted positives, how many were correct. "
                      "Recall = of all true positives, how many were detected. "
                      "F1 = harmonic mean of both — the primary single-number quality metric per class. "
                      "All classes maintain high scores with minimal variance, showing balanced performance.",
            "metric": "Macro Avg F1: 95.79% · All classes above 0.87 F1 · Raw data: clv2_classification_report.csv",
            "path":   "Training/Classifier Performance Data/PrecisionRecallF1.png",
        },
        # Detector plots
        {
            "title":  "TRAINING RESULTS OVERVIEW — YOLOv8 Detector",
            "model":  "Model: YOLOv8n (ONNX/PT) · Task: 17-class fruit detection with bounding boxes",
            "desc":   "Composite plot of all key training and validation metrics across 50 epochs: "
                      "box loss, classification loss, DFL loss, precision, recall, mAP@50, and mAP@50-95. "
                      "All loss curves drop cleanly and all metric curves rise to near-ceiling, confirming "
                      "stable training with no instability or plateaus.",
            "metric": "mAP@50: 98.98% · mAP@50-95: 98.97% · Precision: 97.99% · Recall: 95.34%",
            "path":   "Training/Detector Performance Data/fruit_detector_v158/results.png",
        },
        {
            "title":  "NORMALIZED CONFUSION MATRIX — YOLOv8 Detector",
            "model":  "Model: YOLOv8n (ONNX/PT) · Task: 17-class fruit detection with bounding boxes",
            "desc":   "Row-normalized confusion matrix showing per-class recall rates. "
                      "Each row sums to 1.0, making it easy to compare class-level performance regardless "
                      "of sample size. Values close to 1.0 on the diagonal confirm the detector reliably "
                      "identifies each class. The background column shows missed detection rate.",
            "metric": "Near-perfect diagonal across all classes · Minimal background confusion.",
            "path":   "Training/Detector Performance Data/fruit_detector_v158/confusion_matrix_normalized.png",
        },
        {
            "title":  "PRECISION-RECALL CURVE — YOLOv8 Detector",
            "model":  "Model: YOLOv8n (ONNX/PT) · Task: 17-class fruit detection with bounding boxes",
            "desc":   "Per-class precision-recall curves. Area Under the Curve (AUC) equals Average Precision (AP). "
                      "All curves hug the top-right corner — the ideal shape — meaning the detector maintains "
                      "near-perfect precision even at high recall levels. This is the single most comprehensive "
                      "indicator of detector quality.",
            "metric": "mAP@0.5: 0.990 across all classes · Lowest per-class AP: lemon at 0.976.",
            "path":   "Training/Detector Performance Data/fruit_detector_v158/BoxPR_curve.png",
        },
        {
            "title":  "F1 SCORE CURVE — YOLOv8 Detector",
            "model":  "Model: YOLOv8n (ONNX/PT) · Task: 17-class fruit detection with bounding boxes",
            "desc":   "F1 score plotted against confidence threshold per class. The flat plateau across "
                      "a wide confidence range (0.1-0.9) shows the detector is robust — it doesn't require "
                      "careful threshold tuning to perform well. Peak F1 identifies the optimal deployment threshold.",
            "metric": "All classes 0.97 F1 at 0.693 confidence threshold.",
            "path":   "Training/Detector Performance Data/fruit_detector_v158/BoxF1_curve.png",
        },
        {
            "title":  "PRECISION CURVE — YOLOv8 Detector",
            "model":  "Model: YOLOv8n (ONNX/PT) · Task: 17-class fruit detection with bounding boxes",
            "desc":   "Precision plotted against confidence threshold per class. As confidence threshold "
                      "increases, precision rises toward 1.0 — meaning at high confidence, nearly every "
                      "detection the model makes is correct. Used alongside the F1 curve to select the "
                      "optimal operating point for deployment.",
            "metric": "All classes reach 1.00 precision at high confidence threshold.",
            "path":   "Training/Detector Performance Data/fruit_detector_v158/BoxP_curve.png",
        },
        {
            "title":  "VALIDATION PREDICTIONS — YOLOv8 Detector",
            "model":  "Model: YOLOv8n (ONNX/PT) · Task: 17-class fruit detection with bounding boxes",
            "desc":   "Sample validation batch with predicted bounding boxes and class labels overlaid. "
                      "Tight, accurate boxes around the correct fruit classes confirm the detector localizes "
                      "and identifies objects correctly on unseen validation data. "
                      "High confidence scores across detections indicate a well-calibrated model.",
            "metric": "Val batch 0 — representative sample from the held-out validation set.",
            "path":   "Training/Detector Performance Data/fruit_detector_v158/val_batch0_pred.jpg",
        },
        {
            "title":  "VALIDATION PREDICTIONS (BATCH 2) — YOLOv8 Detector",
            "model":  "Model: YOLOv8n (ONNX/PT) · Task: 17-class fruit detection with bounding boxes",
            "desc":   "Second sample validation batch showing detection performance across a different "
                      "set of images. Demonstrates consistent accuracy across varied fruit types, backgrounds, "
                      "and image compositions in the validation set.",
            "metric": "Val batch 2 — further evidence of generalization across the validation set.",
            "path":   "Training/Detector Performance Data/fruit_detector_v158/val_batch2_pred.jpg",
        },
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Training Results — FruitVisionRT")
        self.configure(fg_color=BG_MAIN)
        self.geometry("900x820")
        self.minsize(700, 500)
        self.ctk_images = []  # keep references so images don't get garbage collected
        self._build_ui()

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color=BG_SURFACE, corner_radius=0,
                              height=52, border_width=1, border_color=BORDER)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        ctk.CTkLabel(header, text="TRAINING RESULTS", font=("Courier New", 15, "bold"),
                     text_color=TEXT).pack(side="left", padx=20, pady=14)
        ctk.CTkLabel(header, text="YOLOv8 + MobileNetV2  —  scroll to explore",
                     font=("Courier New", 10), text_color=TEXT_MUTED).pack(side="right", padx=20)

        scroll = ctk.CTkScrollableFrame(self, fg_color=BG_MAIN, corner_radius=0,
                                        scrollbar_button_color=BORDER,
                                        scrollbar_button_hover_color=GREEN_DIM)
        scroll.grid(row=1, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        for i, section in enumerate(self.SECTIONS):
            self._build_section(scroll, section, i)

    def _build_section(self, parent, section, index):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8,
                            border_width=1, border_color=BORDER)
        card.grid(row=index, column=0, sticky="ew", padx=24, pady=(16, 4))
        card.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(card, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 2))
        ctk.CTkLabel(title_row, text=f"{index+1:02d}", font=("Courier New", 11, "bold"),
                     text_color=GREEN_DIM).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(title_row, text=section["title"], font=("Courier New", 13, "bold"),
                     text_color=TEXT).pack(side="left")

        ctk.CTkFrame(card, fg_color=BORDER, height=1).grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 10))

        ctk.CTkLabel(card, text=section["model"], font=("Courier New", 10),
                     text_color=GREEN).grid(row=2, column=0, sticky="w", padx=16, pady=(0, 6))

        ctk.CTkLabel(card, text=section["desc"], font=("Courier New", 11),
                     text_color=TEXT_MUTED, wraplength=780, justify="left").grid(
            row=3, column=0, sticky="w", padx=16, pady=(0, 6))

        metric_bar = ctk.CTkFrame(card, fg_color=BG_SURFACE, corner_radius=4)
        metric_bar.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 12))
        ctk.CTkLabel(metric_bar, text=f"  >  {section['metric']}", font=("Courier New", 10, "bold"),
                     text_color=TEXT_GREEN).pack(anchor="w", padx=8, pady=6)

        try:
            img = Image.open(section["path"])
            display_w = 820
            display_h = int(img.height * (display_w / img.width))
            img = img.resize((display_w, display_h), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(display_w, display_h))
            self.ctk_images.append(ctk_img)
            ctk.CTkLabel(card, image=ctk_img, text="", fg_color=BG_MAIN,
                         corner_radius=4).grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 16))
        except FileNotFoundError:
            ctk.CTkLabel(card, text=f"Image not found: {section['path']}",
                         font=("Courier New", 11), text_color=AMBER).grid(
                row=5, column=0, padx=16, pady=(0, 16))


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()