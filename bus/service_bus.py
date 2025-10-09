import json
import logging
from typing import Callable, Awaitable
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage
from azure.servicebus.exceptions import ServiceBusError

log = logging.getLogger(__name__)

class BusConsumer:
    def __init__(self, conn_str: str, queue_name: str):
        self.conn_str = conn_str
        self.queue_name = queue_name
        self.client = ServiceBusClient.from_connection_string(conn_str)
        self.processor = None

    def _body_to_bytes(self, body) -> bytes:
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

    async def start(self, on_message: Callable[[dict], Awaitable[None]], max_concurrency: int = 4):
        async with self.client:
            self.processor = self.client.get_queue_receiver(queue_name=self.queue_name)
            async with self.processor:
                log.info("Starting message loop...")
                async for msg in self.processor:
                    try:
                        body_bytes = self._body_to_bytes(msg.body)
                        data = json.loads(body_bytes.decode("utf-8"))
                        await on_message(data)
                        await self.processor.complete_message(msg)
                    except Exception as e:
                        log.exception("Processing failed; dead-lettering message.")
                        await self.processor.dead_letter_message(
                            msg,
                            reason="processing_failed",
                            error_description=str(e)[:4096]
                        )

