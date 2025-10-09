import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

# Tenta carregar YOLO; se não existir, usa fallback OpenCV
_USE_YOLO = True
try:
    from ultralytics import YOLO
except Exception:
    _USE_YOLO = False
    YOLO = None

class PlateDetector:
    def __init__(self, weights_path: Optional[str] = None, conf: float = 0.25, iou: float = 0.45):
        self.use_yolo = _USE_YOLO and weights_path and os.path.exists(weights_path)
        self.model = YOLO(weights_path) if self.use_yolo else None
        self.conf = conf
        self.iou  = iou

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        if self.use_yolo:
            return self._detect_yolo(img_bgr)
        return self._detect_fallback(img_bgr)

    def _detect_yolo(self, img_bgr: np.ndarray) -> List[Dict]:
        # Assume que o modelo tem classe "license-plate" ou similar
        results = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, verbose=False)
        dets: List[Dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": cls_id,
                    "label": r.names.get(cls_id, "plate")
                })
        return dets

    def _detect_fallback(self, img_bgr: np.ndarray) -> List[Dict]:
        """
        Fallback simples: procura regiões retangulares com alta densidade de bordas que
        pareçam placas (aspect ratio ~ 2–6, área mínima).
        Não é tão robusto quanto YOLO, mas evita paradas em produção.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 80, 200)
        edges = cv2.dilate(edges, None, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        dets: List[Dict] = []
        h, w = img_bgr.shape[:2]
        min_area = (w*h) * 0.001  # 0.1% da imagem
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < min_area:
                continue
            ar = cw / max(ch, 1)
            if 1.8 <= ar <= 6.5:  # aspecto típico de placa
                dets.append({
                    "bbox": [x, y, x+cw, y+ch],
                    "conf": 0.40,
                    "class_id": 0,
                    "label": "plate_fallback"
                })
        # NMS simples para reduzir sobreposição
        dets = self._nms(dets, iou_thresh=0.3)
        return dets

    @staticmethod
    def _nms(dets: List[Dict], iou_thresh: float = 0.3) -> List[Dict]:
        if not dets:
            return dets
        boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
        scores = np.array([d["conf"] for d in dets], dtype=np.float32)

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return [dets[i] for i in keep]

    @staticmethod
    def draw_annotations(img_bgr: np.ndarray, dets: List[Dict]) -> np.ndarray:
        annotated = img_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{d.get("label","plate")} {d.get("conf",0):.2f}'
            cv2.putText(annotated, label, (x1, max(0, y1-5)),
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
