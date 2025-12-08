# Kafka Setup Guide

## Khởi động Kafka services

```bash
# Start Kafka ecosystem
docker-compose -f docker-compose.kafka.yml up -d

# Check logs
docker-compose -f docker-compose.kafka.yml logs -f

# Stop Kafka
docker-compose -f docker-compose.kafka.yml down
```

## Kiểm tra Kafka

1. **Kafka UI**: http://localhost:8080

   - Xem topics, messages, consumers
   - Monitor cluster health

2. **Test Producer** (từ app):

```python
from app.services.kafka_service import initialize_kafka, publish_query_event
import asyncio

async def test():
    await initialize_kafka()

    test_event = {
        "query_id": "test-123",
        "user_id": "user-1",
        "query_text": "Test query",
        "faculty": "CNTT",
        "year": "2025",
        "timestamp": "2025-12-07T23:00:00"
    }

    await publish_query_event(test_event)
    print("Event published!")

asyncio.run(test())
```

3. **View Messages** (CLI):

```bash
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic user-queries \
  --from-beginning \
  --max-messages 10
```

## Environment Variables

Add to your `.env`:

```
KAFKA_BROKER_URL=localhost:9092
KAFKA_TOPIC_QUERIES=user-queries
```

## Integration Flow

```
User Query → FastAPI App (localhost)
           ↓
      MongoDB (localhost:27017) - Query Logs ✅
           ↓
      Kafka Producer → Kafka (localhost:9092)
           ↓
      Kafka Topic: user-queries
           ↓
      Spark Streaming (localhost:4040) 🔥 NEW
           ↓
      MongoDB - Analytics (3 collections):
        • query_analytics_by_faculty
        • query_analytics_by_year
        • query_analytics_heatmap
```

**Real-time Analytics**: Spark Streaming xử lý events từ Kafka và tạo aggregations theo faculty, year, và heatmap. Xem [Spark Setup Guide](SPARK_SETUP.md) để biết chi tiết.

## Troubleshooting

**Connection refused**:

- Check Kafka is running: `docker ps | grep kafka`
- Wait 30s after startup for Kafka to be ready
- Check logs: `docker logs kafka`

**Topic not found**:

- Auto-create is enabled, topic will be created on first publish
- Manual create:

```bash
docker exec -it kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic user-queries \
  --partitions 3 \
  --replication-factor 1
```
