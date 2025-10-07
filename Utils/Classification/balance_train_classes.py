import os
import random
from tqdm import tqdm

TRAIN_DIR = r"C:\Users\DM77\Documents\FruitVisionRT\data\classification\Fruit and Vegetables\train"
TARGET_COUNT = 820

def balance_train_classes(train_dir, target_count):
    print(f" Balancing each training class to {target_count} images...\n")
    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        total = len(images)

        if total > target_count:
            print(f" {class_name}: {total} → {target_count} (removing {total - target_count})")
            to_remove = random.sample(images, total - target_count)
            for img in tqdm(to_remove, desc=f"Removing from {class_name}"):
                os.remove(os.path.join(class_path, img))
        else:
            print(f" {class_name}: {total} (no trimming needed)")
    print("\n Train class balancing complete!")

balance_train_classes(TRAIN_DIR, TARGET_COUNT)
