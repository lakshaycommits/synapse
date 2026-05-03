import json
from aiokafka import AIOKafkaProducer
import os
from dotenv import load_dotenv
from ..utils.logger import get_logger
import json
import os
from dotenv import load_dotenv
from aiokafka import AIOKafkaProducer

from ..utils.logger import get_logger

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVER")
TOPIC = os.getenv("KAFKA_INGEST_TOPIC")


class Producer:
    def __init__(self):
        self._logger = get_logger(__name__)
        self._producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    async def start(self):
        await self._producer.start()

    async def stop(self):
        await self._producer.stop()

    def get_producer_instance(self):
        return self._producer

    async def publish_ingest_event(self, file_path: str, filename: str):
        self._logger.info("Publishing event for file: %s at path: %s", filename, file_path)
        await self._producer.send_and_wait(
            TOPIC, value={"file_path": file_path, "filename": filename}
        )
        self._logger.debug("Event published successfully for: %s", filename)
