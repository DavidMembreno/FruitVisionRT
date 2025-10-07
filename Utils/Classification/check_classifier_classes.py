import os

dataset_path = "C:/Users/DM77/Documents/FruitVisionRT/data/classification_v2/train"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Path does not exist: {dataset_path}")

class_names = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

print(f"\nTotal classes: {len(class_names)}")
