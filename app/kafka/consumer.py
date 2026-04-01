import asyncio
import json
import os
from pathlib import Path

from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv

# Move imports OUTSIDE loop
from utils.qdrantClient import qdrantClient
from utils.embeddings import Embeddings
from utils.logger import get_logger
from rag.ingest import Ingestion

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVER")
TOPIC = os.getenv("KAFKA_INGEST_TOPIC")


async def consume():
    # INIT ONCE
    logger = get_logger(__name__)
    logger.info("Initializing Qdrant + Embeddings...")

    qdrant = qdrantClient()
    embeddings = Embeddings()

    logger.info("Initialization complete")

    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="synapse-ingest-group",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    await consumer.start()
    logger.info("Listening on topic: %s", TOPIC)

    try:
        async for msg in consumer:
            logger.debug("Received message: %s", msg.value)

            file_path = msg.value.get("file_path")
            filename = msg.value.get("filename")

            try:
                logger.info("Ingesting %s", filename)

                chunks = Ingestion.ingest(
                    [Path(file_path)],
                    qdrant,
                    embeddings
                )

                logger.info("Ingest succeeded: %s → %s chunks", filename, chunks)

                Path(file_path).unlink(missing_ok=True)

            except Exception as e:
                logger.exception("Error ingesting %s", filename)

    finally:
        logger.info("Closing consumer")
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(consume())
