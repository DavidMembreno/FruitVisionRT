FruitVisionRT – Refined Codebase
================================

How to Run:
-----------
- On Desktop (Windows/Linux):
    Run: python main_final.py
    Requires: models/detector/best.pt and models/classifier/clv2_mobilenetv2_smooth05_v2.tflite

- On Raspberry Pi 4:
    Run: python3 main_final_pi.py
    Requires: same models as above
    GUI will open with radio button mode selection and confidence slider.

Models:
-------
- Detector: models/detector/best.pt   (YOLOv8, trained on LVIS Fruits & Vegetables dataset)
- Classifier: models/classifier/clv2_mobilenetv2_smooth05_v2.tflite   (MobileNetV2, crops from same dataset)

Notes:
------
- See README.md for dataset details and licensing.
- Utils/ contains preprocessing and QA scripts that produced the training data.
- Training/ shows how models were trained.
- TestScripts/ contains early demos, not needed for final system.

