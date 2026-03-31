import asyncio
import json
import os
from pathlib import Path

from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv

# Move imports OUTSIDE loop
from utils.qdrantClient import qdrantClient
from utils.embeddings import Embeddings
from rag.ingest import Ingestion

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVER")
TOPIC = os.getenv("KAFKA_INGEST_TOPIC")


async def consume():
    # INIT ONCE
    print("[INIT] Initializing Qdrant + Embeddings...")

    qdrant = qdrantClient()
    embeddings = Embeddings()

    print("[INIT] Initialization complete")

    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="synapse-ingest-group",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    await consumer.start()
    print(f"[CONSUMER] Listening on topic: {TOPIC}")

    try:
        async for msg in consumer:
            print(f"[CONSUMER] Received: {msg.value}")

            file_path = msg.value.get("file_path")
            filename = msg.value.get("filename")

            try:
                print(f"[PROCESS] Ingesting {filename}")

                chunks = Ingestion.ingest(
                    [Path(file_path)],
                    qdrant,
                    embeddings
                )

                print(f"[SUCCESS] {filename} → {chunks} chunks")

                Path(file_path).unlink(missing_ok=True)

            except Exception as e:
                print(f"[ERROR] {filename} → {e}")

    finally:
        print("[SHUTDOWN] Closing consumer")
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(consume())
