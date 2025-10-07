import os
import xml.etree.ElementTree as ET

# Paths
ANNOTATIONS_DIR = 'data/detection/annotations'
LABELS_DIR = 'data/detection/labels'
CLASS_LIST = [
    'apple', 'banana', 'orange', 'strawberry', 'grapes', 'watermelon', 'lemon',
    'kiwi', 'mango', 'peach', 'pineapple', 'pomegranate', 'tomato', 'avocado',
    'cucumber', 'pear', 'cherry'
]

# Create labels directory if it doesn't exist
os.makedirs(LABELS_DIR, exist_ok=True)

# Loop through all XML files
for file in os.listdir(ANNOTATIONS_DIR):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text.replace('.jpg', '.txt')
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_lines = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_LIST:
            print(f"Skipping unknown class: {name}")
            continue
        class_id = CLASS_LIST.index(name)

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Normalize
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Save label file
    label_path = os.path.join(LABELS_DIR, filename)
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

print("Conversion complete. YOLO labels saved to:", LABELS_DIR)
