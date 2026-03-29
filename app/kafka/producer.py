import json
from aiokafka import AIOKafkaProducer
import os
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVER")
TOPIC = os.getenv("KAFKA_INGEST_TOPIC")

class Producer:
    def __init__(self):
        self._producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )

    async def start(self):
        await self._producer.start()

    async def stop(self):
        await self._producer.stop()
    
    def get_producer_instance(self):
        return self._producer

    async def publish_ingest_event(self, file_path: str, filename: str):
        print(f"Publishing event for file: {filename} at path: {file_path}")
        await self._producer.send_and_wait(
            TOPIC,
            value={"file_path": file_path, "filename": filename}
        )
        print(f"Event published successfully for: {filename}")
