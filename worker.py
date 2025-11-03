import os
import cv2
import json
import asyncio
import logging
import urllib.request
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timezone

from detection.detector import PlateDetector
from azure.servicebus.aio import ServiceBusClient, AutoLockRenewer
from azure.servicebus.exceptions import MessageLockLostError

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("worker-local")

# ----- env -----
load_dotenv()
RESULTS_DIR  = os.getenv("RESULTS_DIR", "./outputs")
MODEL_PATH   = os.getenv("LPR_MODEL_PATH", "runs/detect/train/weights/best.pt")
LPR_CONF     = float(os.getenv("LPR_CONF", "0.25"))
LPR_IOU      = float(os.getenv("LPR_IOU", "0.45"))
SB_CONN      = os.getenv("AZURE_SERVICEBUS_CONNECTION_STRING")
SB_QUEUE     = os.getenv("AZURE_SERVICEBUS_QUEUE", "processar")

# ----- detector -----
detector = PlateDetector(weights_path=MODEL_PATH, conf=LPR_CONF, iou=LPR_IOU)

# ----- helpers -----
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_image(path: str, img_bgr: np.ndarray, quality: int = 95):
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in [".jpg", ".jpeg"] else []
    _, enc = cv2.imencode(ext, img_bgr, params)
    with open(path, "wb") as f:
        f.write(enc.tobytes())

def download_image_http(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Falha ao decodificar imagem baixada.")
    return img

# ----- BUS -----
class BusConsumer:
    def __init__(self, conn_str: str, queue_name: str):
        self.client = ServiceBusClient.from_connection_string(conn_str)
        self.queue_name = queue_name

    @staticmethod
    def _body_to_bytes(body) -> bytes:
        if isinstance(body, (bytes, bytearray, memoryview)):
            return bytes(body)
        try:
            return b"".join(
                part if isinstance(part, (bytes, bytearray, memoryview)) else bytes(part)
                for part in body
            )
        except TypeError:
            if isinstance(body, str):
                return body.encode("utf-8")
            raise

    async def start(self, on_message, max_concurrency: int = 2):
        async with self.client:
            receiver = self.client.get_queue_receiver(
                queue_name=self.queue_name,
                prefetch_count=max(1, max_concurrency * 2),
                max_wait_time=5,
            )
            async with receiver, AutoLockRenewer() as renewer:
                log.info("Starting message loop...")
                async for msg in receiver:
                    renewer.register(receiver, msg, max_lock_renewal_duration=600)
                    try:
                        body_bytes = self._body_to_bytes(msg.body)
                        data = json.loads(body_bytes.decode("utf-8"))
                        await on_message(data)
                        await receiver.complete_message(msg)
                    except MessageLockLostError:
                        log.warning("Message lock lost; will be redelivered.")
                    except Exception as e:
                        log.exception("Processing failed; dead-lettering message.")
                        try:
                            await receiver.dead_letter_message(
                                msg,
                                reason="processing_failed",
                                error_description=str(e)[:4096],
                            )
                        except MessageLockLostError:
                            log.warning("Could not DLQ: lock lost; letting it requeue.")

# ----- handler -----
async def handle_message(msg: dict):
    """
    Espera payload padrão:
    {
      "BlobUrl": "https://.../uploadsvideos/cameras/1/videoteste/frame_0010.jpg",
      "BlobPath": "cameras/1/videoteste/frame_0010.jpg",
      "Container": "uploadsvideos",
      "CameraId": "1",
      "VideoFileName": "videoteste.mp4",
      "FrameFileName": "frame_0010.jpg",
      "CapturedAtUtc": "2025-10-08T13:19:54.0759989+00:00"
    }
    """
    blob_url  = msg.get("BlobUrl") + '?sp=r&amp;st=2025-11-03T11:42:13Z&amp;se=2026-03-14T19:57:13Z&amp;spr=https&amp;sv=2024-11-04&amp;sr=c&amp;sig=M%2FEppzitlKlHCnBpX1kHSu%2FPovfzo848F%2BHojLCCulM%3D'
    blob_path = msg.get("BlobPath")
    camera_id = str(msg.get("CameraId", "unknown"))
    frame_fn  = msg.get("FrameFileName", "frame.jpg")

    # 1) Baixar frame
    img_bgr = download_image_http(blob_url)

    # 2) Detecção somente de regiões com placa
    dets = detector.detect(img_bgr)
    annotated = detector.draw_annotations(img_bgr, dets)
    crops = detector.crop_regions(img_bgr, dets)

    # 3) Montar paths locais baseados em BlobPath
    base_dir   = os.path.join(RESULTS_DIR, os.path.dirname(blob_path or f"cameras/{camera_id}/unknown"))
    base_name  = os.path.splitext(frame_fn)[0]
    json_path  = os.path.join(base_dir, f"{base_name}.json")
    ann_path   = os.path.join(base_dir, f"{base_name}_annotated.jpg")
    crop_fmt   = os.path.join(base_dir, f"{base_name}_lp_{{i}}.jpg")

    # 4) Salvar resultados locais
    out = {
        "source": {
            "blob_url": blob_url,
            "blob_path": blob_path,
            "camera_id": camera_id,
            "video_file": msg.get("VideoFileName"),
            "frame_file": frame_fn,
            "captured_at_utc": msg.get("CapturedAtUtc"),
        },
        "analysis": {
            "detector": "yolov8",
            "num_detections": len(dets),
            "detections": dets  # bbox, conf_det, class_id, label
        },
        "processed_at_utc": iso_now(),
        "version": "1.2.0-local-no-ocr"
    }
    save_json(json_path, out)
    save_image(ann_path, annotated, quality=92)
    for i, crop in enumerate(crops):
        save_image(crop_fmt.format(i=i), crop, quality=95)

    log.info(f"OK: {json_path}")

# ----- main -----
async def main():
    consumer = BusConsumer(SB_CONN, SB_QUEUE)
    await consumer.start(on_message=handle_message, max_concurrency=2)

if __name__ == "__main__":
    asyncio.run(main())
