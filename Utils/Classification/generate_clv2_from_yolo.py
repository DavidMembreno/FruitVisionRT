# generate_clv2_from_yolo.py

import os
import cv2
from pathlib import Path
import shutil


ROOT_DIR = Path("C:/Users/DM77/Documents/FruitVisionRT")
IMAGES_DIR = ROOT_DIR / "data/detection/images"
LABELS_DIR = ROOT_DIR / "data/detection/labels"
SPLITS = {
    "train": ROOT_DIR / "data/detection/train.txt",
    "val": ROOT_DIR / "data/detection/val.txt",
    "test": ROOT_DIR / "data/detection/test.txt"
}
CLASS_NAMES = [
    'apple', 'banana', 'orange', 'strawberry', 'grapes',
    'watermelon', 'lemon', 'kiwi', 'mango', 'pineapple',
    'pomegranate', 'tomato', 'avocado', 'pear', 'cucumber',
    'carrot', 'bell pepper'
]
OUT_DIR = ROOT_DIR / "data/classification_v2"
IMG_SIZE = (128, 128)  # Resize all crops to same size
MIN_BOX_AREA = 100  # Ignore tiny boxes

def normalize_bbox(bbox, img_w, img_h):
    #Convert normalized YOLO bbox to pixel coordinates
    cls_id, x_c, y_c, w, h = bbox
    x_c, y_c, w, h = float(x_c), float(y_c), float(w), float(h)

    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2), int(cls_id)

def prepare_output_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in SPLITS.keys():
        for cls in CLASS_NAMES:
            os.makedirs(OUT_DIR / split / cls, exist_ok=True)

def process_split(split_name, file_list):
    for image_path in file_list:
        image_path = Path(image_path.strip())
        full_image_path = IMAGES_DIR / image_path.name
        label_path = LABELS_DIR / (image_path.stem + ".txt")

        if not label_path.exists() or not full_image_path.exists():
            continue

        img = cv2.imread(str(full_image_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            bbox = line.strip().split()
            if len(bbox) != 5:
                continue

            x1, y1, x2, y2, cls_id = normalize_bbox(bbox, img_w, img_h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                continue

            crop_resized = cv2.resize(crop, IMG_SIZE)
            class_name = CLASS_NAMES[cls_id]
            save_path = OUT_DIR / split_name / class_name / f"{image_path.stem}_{idx}.jpg"
            cv2.imwrite(str(save_path), crop_resized)

def main():
    print("Creating classification_v2 dataset from YOLO labels...")
    prepare_output_dirs()

    for split_name, path in SPLITS.items():
        if not path.exists():
            print(f"Missing split file: {path}")
            continue

        with open(path, "r") as f:
            file_list = f.readlines()
        print(f"Processing {split_name} set with {len(file_list)} images...")
        process_split(split_name, file_list)

    print("Done! classification_v2 dataset created.")

if __name__ == "__main__":
    main()
