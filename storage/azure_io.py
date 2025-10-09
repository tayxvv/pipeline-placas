import io
import logging
from typing import Tuple, Optional, Dict
from azure.storage.blob import BlobServiceClient, ContentSettings

log = logging.getLogger(__name__)

class AzureBlobIO:
    def __init__(self, conn_str: str):
        self.client = BlobServiceClient.from_connection_string(conn_str)

    def parse_url_or_path(self, container: str, blob_path: str, blob_url: Optional[str]) -> Tuple[str, str]:
        if blob_url and "blob.core" in blob_url:
            # confiar no container da mensagem, path até o final
            # (caso a URL esteja pública com SAS você poderia baixar via requests;
            # aqui preferimos baixar autenticado pelo SDK)
            # Ex.: .../uploadsvideos/cameras/1/vid/frame_0010.jpg
            parts = blob_url.split(".net/")[-1].split("/", 1)
            container = parts[0]
            path = parts[1]
            return container, path
        return container, blob_path

    def download_bytes(self, container: str, blob_path: str) -> Tuple[bytes, Dict[str, str]]:
        blob = self.client.get_blob_client(container=container, blob=blob_path)
        stream = io.BytesIO()
        data = blob.download_blob()
        data.readinto(stream)
        props = blob.get_blob_properties()
        meta = props.metadata or {}
        return stream.getvalue(), meta

    def upload_bytes(self, container: str, blob_path: str, data: bytes, content_type: str = "application/octet-stream", metadata: Optional[Dict[str, str]] = None):
        blob = self.client.get_blob_client(container=container, blob=blob_path)
        log.info(f"Uploading to {container}/{blob_path} ({len(data)} bytes)")
        blob.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
            metadata=metadata or {}
        )

    def set_metadata(self, container: str, blob_path: str, metadata: Dict[str, str]):
        blob = self.client.get_blob_client(container=container, blob=blob_path)
        props = blob.get_blob_properties()
        current = props.metadata or {}
        current.update(metadata)
        blob.set_blob_metadata(current)
