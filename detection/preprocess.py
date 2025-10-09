import cv2
import numpy as np

def preprocess_bgr(img_bgr: np.ndarray) -> np.ndarray:
    # 1) redimensiona mantendo razão para ~1280 máx (bom p/ detector)
    h, w = img_bgr.shape[:2]
    scale = 1280 / max(h, w) if max(h, w) > 1280 else 1.0
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 2) equalização adaptativa no canal L (LAB)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3) leve desruído bilateral (preserva bordas)
    img_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    return img_bgr
