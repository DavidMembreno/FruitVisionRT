# main_final_pi.py
# Only for use on Raspberry Pi

import cv2, time, numpy as np, tensorflow as tf, tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import onnxruntime as ort

IMG_SIZE = 128

DETECTION_CLASSES = [
    'apple', 'banana', 'orange', 'strawberry', 'grapes', 'watermelon', 'lemon', 'kiwi',
    'mango', 'peach', 'pineapple', 'pomegranate', 'tomato', 'avocado', 'cucumber', 'pear', 'cherry'
]

CLASSIFIABLE_CLASSES = [
    'apple', 'avocado', 'banana', 'bell pepper', 'carrot', 'cucumber', 'grapes', 'kiwi',
    'lemon', 'mango', 'orange', 'pear', 'pineapple', 'pomegranate', 'strawberry', 'tomato', 'watermelon'
]
CLASSIFIABLE_SET = set(CLASSIFIABLE_CLASSES)

ONNX_MODEL_PATH = "Models/detector/best_pi.onnx"
TFLITE_PATH = "Models/classifier/clv2_mobilenetv2_smooth05_v2.tflite"

# Model loading
yolo_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
yolo_in = yolo_session.get_inputs()[0].name
yolo_outs = [o.name for o in yolo_session.get_outputs()]
print("[ONNX] outputs:", yolo_outs, flush=True)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
t_in = interpreter.get_input_details()
t_out = interpreter.get_output_details()
print(f"[TFLite] input={t_in[0]['shape']} output={t_out[0]['shape']}", flush=True)


# Camera
def _open_cam(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, f = cap.read()
    if not ok or f is None:
        cap.release();
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

# Classifier
def classify_image(image_bgr):
    # Safety: ensure uint8 HWC
    if image_bgr is None or image_bgr.size == 0:
        return "", 0.0
    img = image_bgr
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)
    interpreter.set_tensor(t_in[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(t_out[0]['index'])
    class_id = int(np.argmax(output))
    confidence = float(output[0][class_id])
    return CLASSIFIABLE_CLASSES[class_id], confidence


_printed_shape = False
_printed_range = False


def _prep(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, 0)
    return x

def _sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))
def _softmax(z, axis=0):
    zmax = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - zmax)
    return e / np.sum(e, axis=axis, keepdims=True)




def _boxes_xywh_640_to_xyxy_px(boxes, W, H):
    cx, cy, w, h = boxes
    x1 = (cx - w / 2) * W / 640.0
    y1 = (cy - h / 2) * H / 640.0
    x2 = (cx + w / 2) * W / 640.0
    y2 = (cy + h / 2) * H / 640.0
    return np.stack([x1, y1, x2, y2], axis=0)


def _boxes_xywh_norm_to_xyxy_px(boxes, W, H):
    cx, cy, w, h = boxes
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return np.stack([x1, y1, x2, y2], axis=0)


def _boxes_xyxy_640_to_xyxy_px(boxes, W, H):
    x1, y1, x2, y2 = boxes
    x1 = x1 * W / 640.0
    y1 = y1 * H / 640.0
    x2 = x2 * W / 640.0
    y2 = y2 * H / 640.0
    return np.stack([x1, y1, x2, y2], axis=0)


def _boxes_xyxy_norm_to_xyxy_px(boxes, W, H):
    x1, y1, x2, y2 = boxes
    x1 = x1 * W
    y1 = y1 * H
    x2 = x2 * W
    y2 = y2 * H
    return np.stack([x1, y1, x2, y2], axis=0)


def _clip_and_filter(xyxy, W, H):
    x1 = np.clip(xyxy[0], 0, W - 1)
    y1 = np.clip(xyxy[1], 0, H - 1)
    x2 = np.clip(xyxy[2], 0, W - 1)
    y2 = np.clip(xyxy[3], 0, H - 1)
    good = (x2 > x1) & (y2 > y1) & ((x2 - x1) >= 10) & ((y2 - y1) >= 10)  # Min size filter
    return np.stack([x1, y1, x2, y2], axis=0), good


def safe_crop(frame, x1, y1, x2, y2, margin_frac=0.08, make_square=True, min_size=28):
    H, W = frame.shape[:2]

    bw = max(1, x2 - x1);
    bh = max(1, y2 - y1)
    m = int(max(bw, bh) * margin_frac)
    x1m, y1m, x2m, y2m = x1 - m, y1 - m, x2 + m, y2 + m

    if make_square:
        w2, h2 = x2m - x1m, y2m - y1m
        s = max(w2, h2)
        cx = (x1m + x2m) // 2
        cy = (y1m + y2m) // 2
        x1m = cx - s // 2
        x2m = cx + s // 2
        y1m = cy - s // 2
        y2m = cy + s // 2

    x1m = max(0, min(W - 1, x1m))
    y1m = max(0, min(H - 1, y1m))
    x2m = max(0, min(W - 1, x2m))
    y2m = max(0, min(H - 1, y2m))

    if (x2m - x1m) < min_size or (y2m - y1m) < min_size:
        return None, (0, 0, 0, 0)
    return frame[y1m:y2m, x1m:x2m].copy(), (x1m, y1m, x2m, y2m)


def nms_and_pack(xyxy_px, cls_ids, cls_confs, conf_thresh=0.25, iou_thresh=0.45):
    if len(cls_confs) == 0:
        return []
    x1, y1, x2, y2 = xyxy_px
    boxes_wh = np.column_stack((x1, y1, x2 - x1, y2 - y1)).astype(float)
    scores = cls_confs.astype(float)
    idxs = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores.tolist(), conf_thresh, iou_thresh)
    if len(idxs) == 0:
        return []
    if isinstance(idxs, tuple):
        idxs = idxs[0]
    idxs = np.array(idxs).flatten().tolist()
    out = []
    for i in idxs:
        out.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), int(cls_ids[i]), float(cls_confs[i])))
    return out



def run_yolo(frame, conf_thresh=0.15):  # Lowered default confidence threshold
    global _printed_shape, _printed_range
    H, W = frame.shape[:2]
    x = _prep(frame)
    outs = yolo_session.run(yolo_outs, {yolo_in: x})
    pred = np.array(outs[0])

    if not _printed_shape:
        print("[ONNX] raw shape:", pred.shape, flush=True)
        _printed_shape = True

    # Unbatch
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        return []

    C, A = pred.shape

    if C not in (21, 22, 25, 85) and A not in (21, 22, 25, 85):
        print(f"[WARN] Unexpected pred shape: {pred.shape}", flush=True)
        return []

    if A in (21, 22, 25, 85) and C > 1000:
        pred = pred.T
        C, A = pred.shape

    print(f"[DEBUG] Processing shape: C={C}, A={A}", flush=True)

    if C == 21:  # 17 classes + 4 box coords
        boxes = pred[:4, :]
        logits = pred[4:, :]
        if not _printed_range:
            print(f"[DBG] logits: min={logits.min():.3f} max={logits.max():.3f}", flush=True)
            _printed_range = True
        probs = _sigmoid(logits)
        cls_ids = np.argmax(probs, axis=0)
        cls_confs = probs[cls_ids, np.arange(A)]
    elif C == 22:  # 17 classes + 4 box coords + 1 objectness
        boxes = pred[:4, :]
        obj = _sigmoid(pred[4, :])
        probs = _sigmoid(pred[5:, :])
        cls_ids = np.argmax(probs, axis=0)
        cls_confs = probs[cls_ids, np.arange(A)] * obj
    else:
        # Default handling
        boxes = pred[:4, :]
        if C > 5:
            obj = _sigmoid(pred[4, :])
            probs = _sigmoid(pred[5:, :])
            cls_ids = np.argmax(probs, axis=0)
            cls_confs = probs[cls_ids, np.arange(A)] * obj
        else:
            # No objectness score
            logits = pred[4:, :]
            probs = _sigmoid(logits)
            cls_ids = np.argmax(probs, axis=0)
            cls_confs = probs[cls_ids, np.arange(A)]


    best_results = []
    for kind in ("xywh_norm", "xywh_640", "xyxy_norm", "xyxy_640"):
        try:
            if kind == "xywh_640":
                xyxy = _boxes_xywh_640_to_xyxy_px(boxes, W, H)
            elif kind == "xywh_norm":
                xyxy = _boxes_xywh_norm_to_xyxy_px(boxes, W, H)
            elif kind == "xyxy_640":
                xyxy = _boxes_xyxy_640_to_xyxy_px(boxes, W, H)
            else:
                xyxy = _boxes_xyxy_norm_to_xyxy_px(boxes, W, H)

            xyxy, good = _clip_and_filter(xyxy, W, H)
            idxs = np.where(good & (cls_confs >= conf_thresh))[0]

            if idxs.size:
                packed = nms_and_pack(xyxy[:, idxs], cls_ids[idxs], cls_confs[idxs],
                                      conf_thresh, iou_thresh=0.45)
                if len(packed) > len(best_results):
                    best_results = packed
                    print(f"[BOXDBG] decode={kind} kept={len(packed)}", flush=True)
        except Exception as e:
            print(f"[WARN] decode {kind} failed: {e}", flush=True)
            continue

    return best_results



# UI

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("FruitVisionRT (Fixed)")
        self.root.configure(bg="white")

        main = tk.Frame(root, bg="white")
        main.pack(fill=tk.BOTH, expand=True)

        self.mode = tk.StringVar(value="only_yolo")
        left = tk.Frame(main, bg="white");
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Add confidence threshold control
        tk.Label(left, text="YOLO Confidence:", bg="white").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.15)
        self.conf_scale = tk.Scale(left, from_=0.05, to=0.8, resolution=0.05,
                                   orient=tk.HORIZONTAL, variable=self.conf_var)
        self.conf_scale.pack(anchor=tk.W, fill=tk.X)

        tk.Label(left, text="Mode:", bg="white").pack(anchor=tk.W, pady=(10, 0))
        modes = [
            ("Only YOLO", "only_yolo"),
            ("Only Classifier", "only_cl"),
            ("Classifier Crop", "classifier_crop"),
            ("Mixed Decision", "mixed_decision")
        ]
        for text, value in modes:
            ttk.Radiobutton(left, text=text, variable=self.mode, value=value,
                            command=lambda v=value: print("[DBG] mode ->", v, flush=True)).pack(anchor=tk.W)

        right = tk.Frame(main, bg="white");
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.video_frame = tk.Label(right, borderwidth=2, relief="groove")
        self.video_frame.pack(padx=4, pady=4)

        self.label_output = tk.Label(right, text="", font=("Times New Roman", 14),
                                     bg="white", justify=tk.LEFT, anchor="w")
        self.label_output.pack(pady=(0, 10), anchor="w")

        self.update()

    def update(self):
        frame = get_frame()
        if frame is None:
            ph = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(ph, "Waiting for USB camera...", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            rgb = cv2.cvtColor(ph, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
            self.label_output.config(text="(no frame yet)")
            self.root.after(100, self.update)
            return

        t0 = time.time()
        display_text = ""
        mode = self.mode.get()
        conf_thresh = self.conf_var.get()

        if mode == "only_yolo":
            dets = run_yolo(frame, conf_thresh=conf_thresh)
            print(f"[YOLO] det_count={len(dets)}", flush=True)
            for (x1, y1, x2, y2, cls_id, conf) in dets[:20]:
                label = DETECTION_CLASSES[cls_id] if 0 <= cls_id < len(DETECTION_CLASSES) else f"cls_{cls_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                display_text += f"YOLO: {label} ({conf:.2f})\n"

        elif mode == "only_cl":
            h, w, _ = frame.shape
            size = min(h, w) // 2
            x1 = w // 2 - size // 2;
            y1 = h // 2 - size // 2
            x2 = x1 + size;
            y2 = y1 + size
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                label, conf = classify_image(crop)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                display_text += f"Classifier: {label} ({conf:.2f})\n"

        elif mode == "classifier_crop":
            dets = run_yolo(frame, conf_thresh=conf_thresh)
            print(f"[CROP] yolo_det_count={len(dets)}", flush=True)

            # Process ALL detections, not just classifiable ones
            for (x1, y1, x2, y2, cls_id, conf) in dets[:20]:
                det_label = DETECTION_CLASSES[cls_id] if 0 <= cls_id < len(DETECTION_CLASSES) else f"cls_{cls_id}"

                scrop, (xa, ya, xb, yb) = safe_crop(frame, x1, y1, x2, y2, margin_frac=0.12, make_square=True,
                                                    min_size=28)
                if scrop is None:
                    print(f"[CROP] tiny/invalid crop for det '{det_label}' -> skip", flush=True)
                    continue

                # Always classify the crop, regardless of YOLO label
                cl_label, cl_conf = classify_image(scrop)
                cv2.rectangle(frame, (xa, ya), (xb, yb), (255, 0, 0), 2)
                cv2.putText(frame, f"{cl_label} ({cl_conf:.2f})", (xa, ya - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                display_text += f"Crop: {cl_label} ({cl_conf:.2f}) [YOLO: {det_label}]\n"

        elif mode == "mixed_decision":
            dets = run_yolo(frame, conf_thresh=conf_thresh)
            print(f"[MIX] yolo_det_count={len(dets)}", flush=True)
            for (x1, y1, x2, y2, cls_id, yolo_conf) in dets[:20]:
                yolo_label = DETECTION_CLASSES[cls_id] if 0 <= cls_id < len(DETECTION_CLASSES) else f"cls_{cls_id}"
                scrop, (xa, ya, xb, yb) = safe_crop(frame, x1, y1, x2, y2, margin_frac=0.12, make_square=True,
                                                    min_size=28)

                if scrop is not None:
                    cl_label, cl_conf = classify_image(scrop)
                    if yolo_label in CLASSIFIABLE_SET:
                        if cl_conf > yolo_conf:
                            final_label, final_conf = cl_label, cl_conf
                        else:
                            final_label, final_conf = yolo_label, yolo_conf
                    else:
                        if cl_conf > 0.3:  # Minimum classifier confidence
                            final_label, final_conf = cl_label, cl_conf
                        else:
                            final_label, final_conf = yolo_label, yolo_conf
                else:
                    # Can't crop...use YOLO
                    final_label, final_conf = yolo_label, yolo_conf

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{final_label} ({final_conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                display_text += f"Mixed: {final_label} ({final_conf:.2f})\n"

        fps = 1.0 / max(1e-6, (time.time() - t0))
        cv2.putText(frame, f"FPS: {fps:.1f} | Conf: {conf_thresh:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)
        self.label_output.config(text=display_text.strip())

        self.root.after(10, self.update)

    def on_close(self):
        try:
            if cap is not None and cap.isOpened():
                cap.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
