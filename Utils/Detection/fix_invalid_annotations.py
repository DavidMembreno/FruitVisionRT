import os
import xml.etree.ElementTree as ET
from PIL import Image

#Note: Not initially ran in this project folder but was added in for documentation sake

ANNOTATIONS_DIR = r"C:\Users\DM77\Documents\Detection Data\All_Datasets\Augmented_Detection\annotations"
IMAGES_DIR = r"C:\Users\DM77\Documents\Detection Data\All_Datasets\Augmented_Detection\images"

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def fix_annotation(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    filename = root.find('filename').text
    image_path = os.path.join(IMAGES_DIR, filename)

    if not os.path.exists(image_path):
        print(f"[SKIP] Missing image for {filename}")
        return False

    try:
        img = Image.open(image_path)
        width, height = img.size
    except:
        print(f"[ERROR] Cannot open image {filename}")
        return False

    # Update size in XML
    size_tag = root.find("size")
    if size_tag is None:
        size_tag = ET.SubElement(root, "size")
        ET.SubElement(size_tag, "width").text = str(width)
        ET.SubElement(size_tag, "height").text = str(height)
        ET.SubElement(size_tag, "depth").text = "3"

    else:
        size_tag.find("width").text = str(width)
        size_tag.find("height").text = str(height)
        if size_tag.find("depth") is None:
            ET.SubElement(size_tag, "depth").text = "3"

    fixed_objects = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = clamp(int(float(bndbox.find("xmin").text)), 0, width)
        ymin = clamp(int(float(bndbox.find("ymin").text)), 0, height)
        xmax = clamp(int(float(bndbox.find("xmax").text)), 0, width)
        ymax = clamp(int(float(bndbox.find("ymax").text)), 0, height)

        if xmax <= xmin or ymax <= ymin:
            print(f"[WARNING] Skipping invalid box in {filename}")
            root.remove(obj)
            continue

        bndbox.find("xmin").text = str(xmin)
        bndbox.find("ymin").text = str(ymin)
        bndbox.find("xmax").text = str(xmax)
        bndbox.find("ymax").text = str(ymax)

        fixed_objects.append(obj)

    if not fixed_objects:
        print(f"[REMOVE] No valid boxes in {filename}, skipping save")
        return False

    tree.write(file_path)
    return True

def fix_all_annotations():
    fixed_count = 0
    files = os.listdir(ANNOTATIONS_DIR)
    xml_files = [f for f in files if f.endswith(".xml")]

    for xml_file in xml_files:
        full_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        if fix_annotation(full_path):
            fixed_count += 1

    print(f"\n Fixed {fixed_count} XML annotation files.")

if __name__ == "__main__":
    fix_all_annotations()
