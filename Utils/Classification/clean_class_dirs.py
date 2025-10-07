#This file is important because it allows us to limited the dataset from kaggle to the ones used in detection such that-
#-the fruits in both cla;ssiofication and detection match perfectly

import os
import shutil

# Target class names (from detection)
TARGET_CLASSES = [
    "apple", "banana", "orange", "strawberry", "grapes", "watermelon",
    "lemon", "kiwi", "mango", "peach", "pineapple", "pomegranate",
    "tomato", "avocado", "cucumber", "pear", "cherry"
]

# Classification data path
BASE_PATH = os.path.join("data", "classification", "Fruit and Vegetables")
SETS = ["train", "test"]

# Toggle to actually move the folders or just simulate
DRY_RUN = False  # Change to False to enable folder moving
BACKUP_FOLDER = os.path.join(BASE_PATH, "backup_removed")

# Ensure backup folder exists
if not DRY_RUN:
    os.makedirs(BACKUP_FOLDER, exist_ok=True)

for split in SETS:
    split_path = os.path.join(BASE_PATH, split)
    if not os.path.isdir(split_path):
        continue

    print(f"\nChecking {split_path}...")

    for item in os.listdir(split_path):
        item_path = os.path.join(split_path, item)

        # Normalize folder name
        item_name_normalized = item.strip().lower()

        if item_name_normalized not in TARGET_CLASSES:
            print(f"0  {item} → Not in class list")

            if not DRY_RUN:
                backup_path = os.path.join(BACKUP_FOLDER, f"{split}_{item}")
                shutil.move(item_path, backup_path)
                print(f"Moved to: {backup_path}")
        else:
            print(f"1  {item} → OK")

