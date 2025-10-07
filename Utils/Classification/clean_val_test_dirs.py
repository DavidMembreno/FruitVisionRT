import os
import shutil

VALID_CLASSES = [
    "apple", "avocado", "banana", "cucumber", "grapes", "kiwi", "lemon", "mango",
    "orange", "pear", "pineapple", "pomegranate", "strawberry", "tomato", "watermelon"
]

BASE_DIR = "data/classification/Fruit and Vegetables"
VAL_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "test")

def clean_directory(path, valid_classes):
    print(f"\n Cleaning: {path}")
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            norm_name = folder.strip().lower()
            if norm_name not in valid_classes:
                print(f"0 Removing: {folder}")
                shutil.rmtree(folder_path)
            else:
                print(f"1 Keeping: {folder}")


if os.path.exists(VAL_DIR):
    clean_directory(VAL_DIR, VALID_CLASSES)
else:
    print("Validation directory not found")

