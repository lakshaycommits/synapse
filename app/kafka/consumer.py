import asyncio
import json
import os
from pathlib import Path

from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVER")
TOPIC = os.getenv("KAFKA_INGEST_TOPIC")

async def consume():
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="synapse-ingest-group",
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    await consumer.start()
    print(f"Consumer started, listening on topic: {TOPIC}")
    
    try:
        async for msg in consumer:
            print(f"Received message: {msg.value}")
            file_path = msg.value.get("file_path")
            filename = msg.value.get("filename")
            
            try:
                # run ingestion
                from utils.qdrantClient import qdrantClient
                from utils.embeddings import Embeddings
                from rag.ingest import Ingestion
                
                qdrant = qdrantClient()
                embeddings = Embeddings()
                chunks = Ingestion.ingest([Path(file_path)], qdrant, embeddings)
                print(f"Ingested {chunks} chunks for {filename}")
                
                # cleanup temp file
                Path(file_path).unlink(missing_ok=True)
                
            except Exception as e:
                print(f"Error ingesting {filename}: {e}")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(consume())
