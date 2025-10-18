FruitVisionRT (Refined Codebase)

Real-time fruit detection and classification for desktop (GPU) and Raspberry Pi 4.
This refined project contains only the files that directly contributed to the final system. 
Older versions and unused checkpoints have been excluded for clarity.

**1) Entry Points**
   
Desktop: main_final.py
Raspberry Pi: main_final_pi.py

**3) Models**
   
Detector: models/detector/best.pt
Classifier: models/classifier/clv2_mobilenetv2_smooth05_v2.tflite

Base pretrained model (yolov8n.pt) from the Ultralytics package is included but not used in the final system.

**3) Training**
   
Detector: Training/train_detector.py
Classifier: Training/train_classifier_clv2.ipynb

Detector performance data is under Training/Detector Performance Data/fruit_detector_v158
Classifier performance data is under Training/Classifier Performance Data

**4) Utils (data preparation and QA)**

convert_voc_to_yolo.py – VOC XML to YOLO conversion

fix_invalid_annotations.py – cleans malformed boxes

convert_paths.py – normalizes dataset split paths

generate_clv2_from_yolo.py – builds classification crops from YOLO boxes

clean_class_dirs.py, clean_val_test_dirs.py – ensures unified class set

balance_train_classes.py – balances training classes

check_labels.py, check_yolo_labels.py, check_classifier_classes.py – QA checks

classification_data_prep.ipynb and detection_data_prep.ipynb – dataset prep notebooks

GPU_check.py – verifies CUDA and Torch environment

camera_test.py – OpenCV camera test

convert_yolo_to_tflite.py – experimental export (not used in final system)

**5) Test Scripts (not needed for final system)**

class_main.py – classifier-only demo

yolo_main.py – YOLO-only demo

main.py – early prototype

**6) Data Source**
   
Models were trained on the LVIS Fruits and Vegetables dataset (Kaggle):
https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset

Both YOLOv8 (detector) and MobileNetV2 (classifier) were trained on this dataset.
The classifier dataset was cropped directly from YOLO detection labels.

**7) License**
   
Released under the MIT License. See LICENSE.txt

**9) Contact**

Student: David Membreno
Mentor: Dr. Chang-Shyh Peng
California Lutheran University
