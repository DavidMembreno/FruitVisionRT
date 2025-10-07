import os

# Relative path
image_dir = "data/detection/images"

# Thje files we m,ust fix
txt_files = [
    "data/detection/train.txt",
    "data/detection/val.txt",
    "data/detection/test.txt"
]

for txt_file in txt_files:
    if not os.path.exists(txt_file):
        print(f" File not found: {txt_file}")
        continue

    with open(txt_file, "r") as f:
        lines = f.readlines()

    # Convert t relatve
    new_lines = []
    for line in lines:
        filename = os.path.basename(line.strip())
        new_path = os.path.join(image_dir, filename).replace("\\", "/")
        new_lines.append(new_path + "\n")

    with open(txt_file, "w") as f:
        f.writelines(new_lines)

    print(f"Rewrote: {txt_file} with {len(new_lines)} paths.")
