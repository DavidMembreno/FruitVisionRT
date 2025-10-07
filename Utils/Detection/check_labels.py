import os

# Use the full absolute paths
image_dir = 'C:/Users/DM77/Documents/FruitVisionRT/data/detection/images'
label_dir = 'C:/Users/DM77/Documents/FruitVisionRT/data/detection/labels'

missing = []

for fname in os.listdir(image_dir):
    if fname.endswith('.jpg'):
        label_name = fname.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_name)
        if not os.path.exists(label_path):
            missing.append(label_path)

if missing:
    print("Missing label files:")
    for f in missing:
        print(f)
else:
    print("✅ All label files exist.")
