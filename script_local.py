import os, json, cv2
import numpy as np
from detection.detector import PlateDetector

MODEL_PATH = "runs/detect/train/weights/best.pt"
IMG_PATH = "image.png"
OUT_DIR = "results"

detector = PlateDetector(MODEL_PATH, conf=0.25, iou=0.45)

img = cv2.imread(IMG_PATH)
dets = detector.detect(img)
ann  = detector.draw_annotations(img, dets)
crops = detector.crop_regions(img, dets)

os.makedirs(OUT_DIR, exist_ok=True)
cv2.imwrite(os.path.join(OUT_DIR, "annotated.jpg"), ann)
for i, c in enumerate(crops):
    cv2.imwrite(os.path.join(OUT_DIR, f"crop_{i}.jpg"), c)
with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
    json.dump({"detections": dets}, f, ensure_ascii=False, indent=2)

print("Feito →", OUT_DIR)
