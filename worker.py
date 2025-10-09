import os
import io
import cv2
import asyncio
import logging
import numpy as np
from dotenv import load_dotenv

from utils.helpers import iso_now, to_json_bytes, build_result_paths
from detection.preprocess import preprocess_bgr
from detection.detector import PlateDetector
from storage.azure_io import AzureBlobIO
from bus.service_bus import BusConsumer

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("worker")

# -------- Config --------
load_dotenv()

SB_CONN = os.getenv("AZURE_SERVICEBUS_CONNECTION_STRING")
SB_QUEUE = os.getenv("AZURE_SERVICEBUS_QUEUE", "processar")
BLOB_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

RESULTS_CONTAINER = os.getenv("RESULTS_CONTAINER", "uploadsvideos")
RESULTS_PREFIX    = os.getenv("RESULTS_PREFIX", "results")
SAVE_CROPS        = os.getenv("SAVE_CROPS", "true").lower() == "true"
SAVE_ANNOTATED    = os.getenv("SAVE_ANNOTATED", "true").lower() == "true"

MODEL_PATH = os.getenv("LPR_MODEL_PATH")
LPR_CONF   = float(os.getenv("LPR_CONF", "0.25"))
LPR_IOU    = float(os.getenv("LPR_IOU", "0.45"))

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))

# -------- Services --------
blobio = AzureBlobIO(BLOB_CONN)
detector = PlateDetector(weights_path=MODEL_PATH, conf=LPR_CONF, iou=LPR_IOU)
consumer = BusConsumer(SB_CONN, SB_QUEUE)

# -------- Handler --------
async def handle_message(msg: dict):
    """
    Mensagem esperada (exemplo):
    {
      "BlobUrl": "https://uploadsvideos.blob.core.windows.net/uploadsvideos/cameras/1/videoteste/frame_0010.jpg",
      "BlobPath": "cameras/1/videoteste/frame_0010.jpg",
      "Container": "uploadsvideos",
      "CameraId": "1",
      "VideoFileName": "videoteste.mp4",
      "FrameFileName": "frame_0010.jpg",
      "CapturedAtUtc": "2025-10-08T13:19:54.0759989+00:00"
    }
    """
    container = msg.get("Container", "")
    blob_path = msg.get("BlobPath", "")
    blob_url  = msg.get("BlobUrl", None)

    # Resolve container/path a partir de url/path
    container, blob_path = blobio.parse_url_or_path(container, blob_path, blob_url)
    log.info(f"Processing frame: {container}/{blob_path}")

    # Baixar bytes + metadata (para poder marcar processed=yes depois)
    img_bytes, meta = blobio.download_bytes(container, blob_path)

    # Decodificar imagem
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Falha ao decodificar imagem (cv2.imdecode retornou None).")

    # Pré-processamento
    img_prep = preprocess_bgr(img_bgr)

    # Detecção
    dets = detector.detect(img_prep)

    # Anotações + crops
    annotated_bgr = detector.draw_annotations(img_prep, dets) if SAVE_ANNOTATED else None
    crops = detector.crop_regions(img_prep, dets) if SAVE_CROPS else []

    # Paths para salvar resultado
    json_path, ann_path, crop_fmt = build_result_paths(blob_path, RESULTS_PREFIX)

    # Monta JSON de saída
    result = {
        "source": {
            "container": container,
            "blob_path": blob_path,
            "camera_id": msg.get("CameraId"),
            "video_file": msg.get("VideoFileName"),
            "frame_file": msg.get("FrameFileName"),
            "captured_at_utc": msg.get("CapturedAtUtc"),
        },
        "analysis": {
            "detector": "yolo" if detector.use_yolo else "opencv_fallback",
            "num_detections": len(dets),
            "detections": dets
        },
        "processed_at_utc": iso_now(),
        "version": "1.0.0"
    }

    # Upload JSON
    blobio.upload_bytes(
        RESULTS_CONTAINER, json_path, to_json_bytes(result),
        content_type="application/json",
        metadata={"type": "license_plate_detections"}
    )

    # Upload Annotated
    if annotated_bgr is not None:
        _, enc = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        blobio.upload_bytes(
            RESULTS_CONTAINER, ann_path, enc.tobytes(),
            content_type="image/jpeg",
            metadata={"annotated": "yes"}
        )

    # Upload crops
    for i, crop in enumerate(crops):
        _, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_path = crop_fmt.format(i=i)
        blobio.upload_bytes(
            RESULTS_CONTAINER, crop_path, enc.tobytes(),
            content_type="image/jpeg",
            metadata={"kind": "plate_crop", "index": str(i)}
        )

    # Marcar o blob de origem como processado (opcional, ajuda na idempotência)
    try:
        blobio.set_metadata(container, blob_path, {"processed": "yes"})
    except Exception:
        log.warning("Não foi possível atualizar metadata do blob de origem.", exc_info=True)

    log.info(f"Done: {container}/{blob_path} -> {RESULTS_CONTAINER}/{json_path}")

# -------- Main --------
async def main():
    await consumer.start(on_message=handle_message, max_concurrency=MAX_CONCURRENCY)

if __name__ == "__main__":
    asyncio.run(main())
