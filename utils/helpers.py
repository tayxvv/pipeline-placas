import os
import re
import orjson
from datetime import datetime, timezone
from typing import Dict

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def to_json_bytes(d: Dict) -> bytes:
    return orjson.dumps(d, option=orjson.OPT_INDENT_2)

def sanitize_path(path: str) -> str:
    # normaliza separadores e remove // duplicados
    return re.sub(r'/{2,}', '/', path.replace('\\', '/'))

def build_result_paths(blob_path: str, results_prefix: str = "results"):
    """
    Ex.: cameras/1/videoteste/frame_0010.jpg  ->
        results/cameras/1/videoteste/frame_0010.json
        results/cameras/1/videoteste/frame_0010_annotated.jpg
        results/cameras/1/videoteste/frame_0010_lp_0.jpg
    """
    blob_path = sanitize_path(blob_path)
    dir_ = os.path.dirname(blob_path)
    base, ext = os.path.splitext(os.path.basename(blob_path))
    json_path = f"{results_prefix}/{dir_}/{base}.json"
    ann_path  = f"{results_prefix}/{dir_}/{base}_annotated{ext or '.jpg'}"
    crop_fmt  = f"{results_prefix}/{dir_}/{base}_lp_{{i}}.jpg"
    return sanitize_path(json_path), sanitize_path(ann_path), crop_fmt

def already_processed(metadata: dict) -> bool:
    # você pode marcar no blob original uma metadata "processed=yes"
    # ou usar outra estratégia (como procurar o JSON de resultado)
    return metadata.get("processed", "no").lower() == "yes"
