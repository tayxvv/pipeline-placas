import cv2
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, weights_path: str, conf: float = 0.25, iou: float = 0.45):
        if not weights_path:
            raise ValueError("Defina LPR_MODEL_PATH com o caminho do modelo YOLO (.pt).")
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou  = iou

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        """Retorna lista de detecções: [{bbox:[x1,y1,x2,y2], conf_det, class_id, label}]"""
        results = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, verbose=False)
        dets: List[Dict] = []
        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            names = getattr(r, "names", {}) or {}
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                box_conf = float(b.conf[0])
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                label = names.get(cls_id, "plate")
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf_det": box_conf,
                    "class_id": cls_id,
                    "label": label,
                })
        return dets

    @staticmethod
    def draw_annotations(img_bgr: np.ndarray, dets: List[Dict]) -> np.ndarray:
        annotated = img_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f'{d.get("label","plate")} ({d.get("conf_det",0):.2f})'
            cv2.putText(annotated, txt, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        return annotated

    @staticmethod
    def crop_regions(img_bgr: np.ndarray, dets: List[Dict]) -> List[np.ndarray]:
        crops = []
        h, w = img_bgr.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
            crops.append(img_bgr[y1:y2, x1:x2].copy())
        return crops
