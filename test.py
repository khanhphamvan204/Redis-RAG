#!/usr/bin/env python3
"""
Send test message directly to Kafka
Run: python send_test_message.py
"""
import json
from kafka import KafkaProducer
from datetime import datetime
import time

# Kafka config
KAFKA_BROKER = "localhost:9092"
TOPIC = "user-queries"

print("=" * 80)
print("SENDING TEST MESSAGE TO KAFKA")
print("=" * 80)

# Create producer
print(f"\nConnecting to Kafka: {KAFKA_BROKER}")
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Test message (matching your app's schema)
test_messages = [
    {
        "query_id": f"test-{int(time.time())}-1",
        "timestamp": datetime.utcnow().isoformat(),
        "user": {
            "user_id": 101,
            "user_type": "Sinh viên",
            "department_id": 1,
            "code": "SV001",
            "years": 3
        },
        "session_id": "test-session-1",
        "query_text": "What is machine learning?",
        "rewritten_query": None,
        "rag_params": {
            "k": 5,
            "similarity_threshold": 0.7,
            "context_found": 3
        },
        "rag_metrics": {
            "response_time_ms": 150.5,
            "answer_length": 200,
            "answer_text": "Machine learning is...",
            "success": True
        },
        "metadata": {
            "model_used": "gemini-2.5-flash",
            "history_used": False,
            "history_count": 0,
            "query_rewritten": False
        }
    },
    {
        "query_id": f"test-{int(time.time())}-2",
        "timestamp": datetime.utcnow().isoformat(),
        "user": {
            "user_id": 102,
            "user_type": "Giáo viên",
            "department_id": 2,
            "code": "GV001",
            "years": None
        },
        "session_id": "test-session-2",
        "query_text": "Explain deep learning",
        "rewritten_query": None,
        "rag_params": {
            "k": 5,
            "similarity_threshold": 0.7,
            "context_found": 4
        },
        "rag_metrics": {
            "response_time_ms": 200.3,
            "answer_length": 300,
            "answer_text": "Deep learning is...",
            "success": True
        },
        "metadata": {
            "model_used": "gemini-2.5-flash",
            "history_used": True,
            "history_count": 2,
            "query_rewritten": True
        }
    },
    {
        "query_id": f"test-{int(time.time())}-3",
        "timestamp": datetime.utcnow().isoformat(),
        "user": {
            "user_id": 103,
            "user_type": "Sinh viên",
            "department_id": 1,
            "code": "SV002",
            "years": 2
        },
        "session_id": "test-session-3",
        "query_text": "What is reinforcement learning?",
        "rewritten_query": None,
        "rag_params": {
            "k": 5,
            "similarity_threshold": 0.7,
            "context_found": 2
        },
        "rag_metrics": {
            "response_time_ms": 180.7,
            "answer_length": 250,
            "answer_text": "Reinforcement learning is...",
            "success": True
        },
        "metadata": {
            "model_used": "gemini-2.5-flash",
            "history_used": False,
            "history_count": 0,
            "query_rewritten": False
        }
    }
]

# Send messages
print(f"\nSending {len(test_messages)} test messages to topic '{TOPIC}'...")
for i, msg in enumerate(test_messages, 1):
    future = producer.send(TOPIC, value=msg, key=msg['query_id'].encode('utf-8'))
    result = future.get(timeout=10)
    print(f"✓ Message {i} sent - Partition: {result.partition}, Offset: {result.offset}")
    time.sleep(0.5)  # Small delay between messages

producer.flush()
producer.close()

print("\n" + "=" * 80)
print("✓ ALL TEST MESSAGES SENT SUCCESSFULLY")
print("=" * 80)
print("\nNow check Spark logs:")
print("  docker logs spark-streaming -f")
print("\nYou should see:")
print("  [faculty] Batch X has 3 records")
print("  [year] Batch X has 2 records (only students)")
print("=" * 80)

