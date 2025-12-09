import logging
import json
import asyncio
from typing import Optional, Dict
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
import os

logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
KAFKA_TOPIC_QUERIES = os.getenv("KAFKA_TOPIC_QUERIES", "user-queries")

# Singleton producer instance
_kafka_producer: Optional[AIOKafkaProducer] = None


async def initialize_kafka():
    """
    Initialize Kafka producer on application startup
    Creates topic if it doesn't exist
    """
    global _kafka_producer
    
    try:
        logger.info(f"Initializing Kafka producer... Broker: {KAFKA_BROKER_URL}")
        
        _kafka_producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BROKER_URL,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='gzip',
            max_request_size=1048576,  # 1MB
            request_timeout_ms=30000,
            retry_backoff_ms=100
        )
        
        await _kafka_producer.start()
        logger.info(f"✓ Kafka producer started successfully - Topic: {KAFKA_TOPIC_QUERIES}")
        
    except Exception as e:
        logger.error(f"Failed to initialize Kafka producer: {e}")
        _kafka_producer = None
        logger.warning("Application will continue without Kafka integration")


async def shutdown_kafka():
    """
    Gracefully shutdown Kafka producer
    """
    global _kafka_producer
    
    if _kafka_producer:
        try:
            await _kafka_producer.stop()
            logger.info("Kafka producer stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Kafka producer: {e}")
        finally:
            _kafka_producer = None


def get_kafka_producer() -> Optional[AIOKafkaProducer]:
    """
    Get the singleton Kafka producer instance
    
    Returns:
        AIOKafkaProducer instance or None if not initialized
    """
    return _kafka_producer


async def publish_query_event(query_log: Dict) -> bool:
    """
    Publish query event to Kafka topic
    
    Args:
        query_log: Dictionary containing query log data
        
    Returns:
        True if published successfully, False otherwise
    """
    producer = get_kafka_producer()
    
    if not producer:
        logger.warning("Kafka producer not available, skipping event publication")
        return False
    
    try:
        # Convert datetime and ObjectId objects to serializable format
        serializable_log = query_log.copy()
        
        # Convert datetime to ISO format
        if 'timestamp' in serializable_log:
            from datetime import datetime
            if isinstance(serializable_log['timestamp'], datetime):
                serializable_log['timestamp'] = serializable_log['timestamp'].isoformat()
        
        # Convert MongoDB ObjectId to string
        if '_id' in serializable_log:
            from bson import ObjectId
            if isinstance(serializable_log['_id'], ObjectId):
                serializable_log['_id'] = str(serializable_log['_id'])
        
        # Log message structure for debugging
        logger.info(f"Publishing to Kafka - Query ID: {serializable_log.get('query_id', 'unknown')[:8]}...")
        logger.debug(f"Message structure: {json.dumps(serializable_log, indent=2)[:500]}...")
        
        # Send message to Kafka
        await producer.send_and_wait(
            topic=KAFKA_TOPIC_QUERIES,
            value=serializable_log,
            key=query_log.get('query_id', '').encode('utf-8') if query_log.get('query_id') else None
        )
        
        logger.info(f"✓ Published query event to Kafka: {query_log.get('query_id', 'unknown')[:8]}...")
        return True
        
    except KafkaError as e:
        logger.error(f"Kafka error publishing event: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error publishing to Kafka: {e}", exc_info=True)
        return False


async def check_kafka_health() -> bool:
    """
    Check if Kafka producer is healthy and ready
    
    Returns:
        True if healthy, False otherwise
    """
    producer = get_kafka_producer()
    return producer is not None and producer._closed is False