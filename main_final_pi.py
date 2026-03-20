# main_final_pi.py

# Raspberry Pi entry point — uses ONNX for detection and TFLite for classification

# Requires a USB camera connected via /dev/video*



import cv2

import time

import numpy as np

import tensorflow as tf

import tkinter as tk

from tkinter import ttk

from PIL import Image, ImageTk

import onnxruntime as ort



CLASSIFIER_INPUT_SIZE = 128



YOLO_CLASSES = [

    'apple', 'banana', 'orange', 'strawberry', 'grapes', 'watermelon', 'lemon', 'kiwi',

    'mango', 'peach', 'pineapple', 'pomegranate', 'tomato', 'avocado', 'cucumber', 'pear', 'cherry'

]



CLASSIFIER_CLASSES = [

    'apple', 'avocado', 'banana', 'bell pepper', 'carrot', 'cucumber', 'grapes', 'kiwi',

    'lemon', 'mango', 'orange', 'pear', 'pineapple', 'pomegranate', 'strawberry', 'tomato', 'watermelon'

]

CLASSIFIABLE_SET = set(CLASSIFIER_CLASSES)



ONNX_MODEL_PATH = "models/best_pi.onnx"

TFLITE_PATH = "models/clv2_mobilenetv2_smooth05_v2.tflite"





# Load models

yolo_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])

yolo_input_name = yolo_session.get_inputs()[0].name

yolo_output_names = [o.name for o in yolo_session.get_outputs()]

print(f"[ONNX] outputs: {yolo_output_names}", flush=True)



interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)

interpreter.allocate_tensors()

tflite_in = interpreter.get_input_details()

tflite_out = interpreter.get_output_details()

print(f"[TFLite] input={tflite_in[0]['shape']} output={tflite_out[0]['shape']}", flush=True)





# Camera — tries common USB video devices in order

def _open_cam(dev):

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

    if not cap.isOpened():

        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.set(cv2.CAP_PROP_FPS, 30)

    ok, frame = cap.read()

    if not ok or frame is None:

        cap.release()

        return None

    return cap





cap = None

for dev in ("/dev/video2", "/dev/video0", "/dev/video1", 2, 0, 1):

    cap = _open_cam(dev)

    if cap is not None:

        print(f"[CAM] Using {dev}", flush=True)

        break

if cap is None:

    print("[ERR] No usable USB camera found.", flush=True)





def get_frame():

    if cap is None or not cap.isOpened():

        return None

    ok, frame = cap.read()

    return frame if ok else None





def classify_image(image_bgr):

    if image_bgr is None or image_bgr.size == 0:

        return "", 0.0

    img = image_bgr if image_bgr.dtype == np.uint8 else image_bgr.astype(np.uint8)

    resized = cv2.resize(img, (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), interpolation=cv2.INTER_AREA)

    normalized = resized.astype(np.float32) / 255.0

    tensor = np.expand_dims(normalized, axis=0)

    interpreter.set_tensor(tflite_in[0]['index'], tensor)

    interpreter.invoke()

    output = interpreter.get_tensor(tflite_out[0]['index'])

    class_id = int(np.argmax(output))

    confidence = float(output[0][class_id])

    return CLASSIFIER_CLASSES[class_id], confidence





# ONNX preprocessing

def _prep_frame(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)

    x = resized.astype(np.float32) / 255.0

    x = np.transpose(x, (2, 0, 1))

    return np.expand_dims(x, 0)





def _sigmoid(z):

    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))





# Box format converters — ONNX exports vary, so we try all formats and keep the best result

def _xywh_640_to_xyxy(boxes, W, H):

    cx, cy, w, h = boxes

    return np.stack([(cx - w/2) * W/640, (cy - h/2) * H/640,

                     (cx + w/2) * W/640, (cy + h/2) * H/640], axis=0)



def _xywh_norm_to_xyxy(boxes, W, H):

    cx, cy, w, h = boxes

    return np.stack([(cx - w/2) * W, (cy - h/2) * H,

                     (cx + w/2) * W, (cy + h/2) * H], axis=0)



def _xyxy_640_to_px(boxes, W, H):

    x1, y1, x2, y2 = boxes

    return np.stack([x1*W/640, y1*H/640, x2*W/640, y2*H/640], axis=0)



def _xyxy_norm_to_px(boxes, W, H):

    x1, y1, x2, y2 = boxes

    return np.stack([x1*W, y1*H, x2*W, y2*H], axis=0)





def _clip_boxes(xyxy, W, H):

    x1 = np.clip(xyxy[0], 0, W - 1)

    y1 = np.clip(xyxy[1], 0, H - 1)

    x2 = np.clip(xyxy[2], 0, W - 1)

    y2 = np.clip(xyxy[3], 0, H - 1)

    valid = (x2 > x1) & (y2 > y1) & ((x2 - x1) >= 10) & ((y2 - y1) >= 10)

    return np.stack([x1, y1, x2, y2], axis=0), valid





def safe_crop(frame, x1, y1, x2, y2, margin_frac=0.08, make_square=True, min_size=28):

    H, W = frame.shape[:2]

    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    m = int(max(bw, bh) * margin_frac)

    x1m, y1m, x2m, y2m = x1 - m, y1 - m, x2 + m, y2 + m



    if make_square:

        s = max(x2m - x1m, y2m - y1m)

        cx, cy = (x1m + x2m) // 2, (y1m + y2m) // 2

        x1m, x2m = cx - s//2, cx + s//2

        y1m, y2m = cy - s//2, cy + s//2



    x1m = max(0, min(W - 1, x1m))

    y1m = max(0, min(H - 1, y1m))

    x2m = max(0, min(W - 1, x2m))

    y2m = max(0, min(H - 1, y2m))



    if (x2m - x1m) < min_size or (y2m - y1m) < min_size:

        return None, (0, 0, 0, 0)

    return frame[y1m:y2m, x1m:x2m].copy(), (x1m, y1m, x2m, y2m)





def nms_and_pack(xyxy, cls_ids, cls_confs, conf_thresh=0.25, iou_thresh=0.45):

    if len(cls_confs) == 0:

        return []

    x1, y1, x2, y2 = xyxy

    boxes_wh = np.column_stack((x1, y1, x2 - x1, y2 - y1)).astype(float)

    idxs = cv2.dnn.NMSBoxes(boxes_wh.tolist(), cls_confs.astype(float).tolist(), conf_thresh, iou_thresh)

    if len(idxs) == 0:

        return []

    if isinstance(idxs, tuple):

        idxs = idxs[0]

    idxs = np.array(idxs).flatten().tolist()

    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), int(cls_ids[i]), float(cls_confs[i])) for i in idxs]





# One-time debug flags so we don't spam the console every frame

_logged_shape = False

_logged_range = False





def run_yolo(frame, conf_thresh=0.15):

    global _logged_shape, _logged_range

    H, W = frame.shape[:2]

    x = _prep_frame(frame)

    raw = np.array(yolo_session.run(yolo_output_names, {yolo_input_name: x})[0])



    if not _logged_shape:

        print(f"[ONNX] raw output shape: {raw.shape}", flush=True)

        _logged_shape = True



    if raw.ndim == 3 and raw.shape[0] == 1:

        raw = raw[0]

    if raw.ndim != 2:

        return []



    C, A = raw.shape



    # Transpose if anchors are in the first dimension

    if A in (21, 22, 25, 85) and C > 1000:

        raw = raw.T

        C, A = raw.shape



    if C not in (21, 22, 25, 85) and A not in (21, 22, 25, 85):

        print(f"[WARN] Unexpected output shape: C={C} A={A}", flush=True)

        return []



    print(f"[ONNX] decoding: C={C} A={A}", flush=True)



    boxes = raw[:4, :]

    if C == 21:

        # 4 box + 17 class logits, no objectness

        logits = raw[4:, :]

        if not _logged_range:

            print(f"[ONNX] logit range: {logits.min():.3f} to {logits.max():.3f}", flush=True)

            _logged_range = True

        probs = _sigmoid(logits)

        cls_ids = np.argmax(probs, axis=0)

        cls_confs = probs[cls_ids, np.arange(A)]

    elif C == 22:

        # 4 box + 1 objectness + 17 class logits

        obj = _sigmoid(raw[4, :])

        probs = _sigmoid(raw[5:, :])

        cls_ids = np.argmax(probs, axis=0)

        cls_confs = probs[cls_ids, np.arange(A)] * obj

    else:

        if C > 5:

            obj = _sigmoid(raw[4, :])

            probs = _sigmoid(raw[5:, :])

            cls_ids = np.argmax(probs, axis=0)

            cls_confs = probs[cls_ids, np.arange(A)] * obj

        else:

            logits = raw[4:, :]

            probs = _sigmoid(logits)

            cls_ids = np.argmax(probs, axis=0)

            cls_confs = probs[cls_ids, np.arange(A)]



    # Try all box formats, keep whichever gives the most detections

    best = []

    converters = {

        "xywh_norm": _xywh_norm_to_xyxy,

        "xywh_640":  _xywh_640_to_xyxy,

        "xyxy_norm": _xyxy_norm_to_px,

        "xyxy_640":  _xyxy_640_to_px,

    }

    for name, convert in converters.items():

        try:

            xyxy = convert(boxes, W, H)

            xyxy, valid = _clip_boxes(xyxy, W, H)

            keep = np.where(valid & (cls_confs >= conf_thresh))[0]

            if keep.size:

                packed = nms_and_pack(xyxy[:, keep], cls_ids[keep], cls_confs[keep], conf_thresh)

                if len(packed) > len(best):

                    best = packed

                    print(f"[ONNX] best decode so far: {name} kept={len(packed)}", flush=True)

        except Exception as e:

            print(f"[WARN] decode {name} failed: {e}", flush=True)



    return best





class App:

    def __init__(self, root):

        self.root = root

        self.root.title("FruitVisionRT — Pi")

        self.root.configure(bg="white")



        self.mode = tk.StringVar(value="only_yolo")

        self.conf_var = tk.DoubleVar(value=0.15)



        main = tk.Frame(root, bg="white")

        main.pack(fill=tk.BOTH, expand=True)



        self._build_controls(main)

        self._build_feed(main)

        self.update()



    def _build_controls(self, parent):

        left = tk.Frame(parent, bg="white")

        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)



        tk.Label(left, text="Confidence:", bg="white").pack(anchor=tk.W)

        tk.Scale(left, from_=0.05, to=0.8, resolution=0.05, orient=tk.HORIZONTAL,

                 variable=self.conf_var).pack(anchor=tk.W, fill=tk.X)



        tk.Label(left, text="Mode:", bg="white").pack(anchor=tk.W, pady=(10, 0))

        modes = [

            ("Only YOLO",       "only_yolo"),

            ("Only Classifier", "only_cl"),

            ("Classifier Crop", "classifier_crop"),

            ("Mixed Decision",  "mixed_decision"),

        ]

        for label, value in modes:

            ttk.Radiobutton(left, text=label, variable=self.mode, value=value).pack(anchor=tk.W)



    def _build_feed(self, parent):

        right = tk.Frame(parent, bg="white")

        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)



        self.video_label = tk.Label(right, borderwidth=2, relief="groove")

        self.video_label.pack(padx=4, pady=4)



        self.text_label = tk.Label(right, text="", font=("Courier New", 12),

                                   bg="white", justify=tk.LEFT, anchor="w")

        self.text_label.pack(pady=(0, 10), anchor="w")



    def update(self):

        frame = get_frame()

        if frame is None:

            placeholder = np.zeros((240, 320, 3), dtype=np.uint8)

            cv2.putText(placeholder, "Waiting for camera...", (10, 120),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self._show_frame(placeholder)

            self.text_label.config(text="No camera feed")

            self.root.after(100, self.update)

            return



        t0 = time.time()

        output_text = ""

        mode = self.mode.get()

        conf = self.conf_var.get()



        if mode == "only_yolo":

            dets = run_yolo(frame, conf_thresh=conf)

            for (x1, y1, x2, y2, cls_id, score) in dets[:20]:

                label = YOLO_CLASSES[cls_id] if 0 <= cls_id < len(YOLO_CLASSES) else f"cls_{cls_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                output_text += f"YOLO: {label} ({score:.2f})\n"



        elif mode == "only_cl":

            h, w, _ = frame.shape

            size = min(h, w) // 2

            x1 = w//2 - size//2

            y1 = h//2 - size//2

            x2, y2 = x1 + size, y1 + size

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:

                label, score = classify_image(crop)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                output_text += f"Classifier: {label} ({score:.2f})\n"



        elif mode == "classifier_crop":

            dets = run_yolo(frame, conf_thresh=conf)

            for (x1, y1, x2, y2, cls_id, score) in dets[:20]:

                det_label = YOLO_CLASSES[cls_id] if 0 <= cls_id < len(YOLO_CLASSES) else f"cls_{cls_id}"

                crop, (xa, ya, xb, yb) = safe_crop(frame, x1, y1, x2, y2, margin_frac=0.12, min_size=28)

                if crop is None:

                    continue

                cl_label, cl_score = classify_image(crop)

                cv2.rectangle(frame, (xa, ya), (xb, yb), (255, 0, 0), 2)

                cv2.putText(frame, f"{cl_label} {cl_score:.2f}", (xa, ya - 10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                output_text += f"Crop: {cl_label} ({cl_score:.2f}) [YOLO: {det_label}]\n"



        elif mode == "mixed_decision":

            dets = run_yolo(frame, conf_thresh=conf)

            for (x1, y1, x2, y2, cls_id, yolo_score) in dets[:20]:

                yolo_label = YOLO_CLASSES[cls_id] if 0 <= cls_id < len(YOLO_CLASSES) else f"cls_{cls_id}"

                crop, (xa, ya, xb, yb) = safe_crop(frame, x1, y1, x2, y2, margin_frac=0.12, min_size=28)



                if crop is not None:

                    cl_label, cl_score = classify_image(crop)

                    if yolo_label in CLASSIFIABLE_SET:

                        final_label, final_score = (cl_label, cl_score) if cl_score > yolo_score else (yolo_label, yolo_score)

                    else:

                        # YOLO detected something the classifier wasn't trained on — use classifier if confident

                        final_label, final_score = (cl_label, cl_score) if cl_score > 0.3 else (yolo_label, yolo_score)

                else:

                    final_label, final_score = yolo_label, yolo_score



                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                cv2.putText(frame, f"{final_label} {final_score:.2f}", (x1, y1 - 10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                output_text += f"Mixed: {final_label} ({final_score:.2f})\n"



        fps = 1.0 / max(1e-6, time.time() - t0)

        cv2.putText(frame, f"FPS: {fps:.1f} | Conf: {conf:.2f}", (10, 30),

                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



        self._show_frame(frame)

        self.text_label.config(text=output_text.strip())

        self.root.after(10, self.update)



    def _show_frame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.video_label.imgtk = imgtk

        self.video_label.configure(image=imgtk)



    def on_close(self):

        if cap is not None and cap.isOpened():

            cap.release()

        self.root.destroy()





if __name__ == "__main__":

    root = tk.Tk()

    app = App(root)

    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.mainloop()

