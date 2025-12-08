# 🚀 Start Kafka + Spark Analytics Pipeline

## Vấn đề hiện tại:

- ✅ Query tracking vào MongoDB: **OK**
- ❌ Kafka không chạy → Queries không được publish
- ❌ Spark không nhận data → Collections analytics **rỗng**

## 🎯 Giải pháp:

### Bước 1: Start Kafka + Spark Stack

```bash
cd "E:\HK1 (2025 - 2026)\BigData\rag\redis_rag"

# Start: Zookeeper, Kafka, Kafka-UI, Spark Streaming
docker-compose -f docker-compose.kafka.yml up -d

# Wait ~30 seconds
timeout /t 30

# Check containers
docker-compose -f docker-compose.kafka.yml ps
```

**Expected output**: 4 containers running

- zookeeper
- kafka
- kafka-ui
- spark-streaming

### Bước 2: Verify Kafka is Ready

```bash
# Check Kafka logs
docker logs kafka --tail 20

# Should see: "started (kafka.server.KafkaServer)"

# Check Spark logs
docker logs spark-streaming --tail 30

# Should see:
# - "Starting Spark Streaming Job..."
# - "Faculty aggregation stream started"
# - "Year aggregation stream started"
# - "Heatmap aggregation stream started"
```

### Bước 3: Generate Test Queries

**Option A: Use Chat UI** (sau khi Kafka chạy)

```
1. Go to http://localhost:3000
2. Login
3. Ask a few questions in chat
→ Queries sẽ tự động publish lên Kafka
```

**Option B: Manual Kafka Test**

```bash
# Publish test message directly to Kafka
docker exec kafka kafka-console-producer --bootstrap-server localhost:9092 --topic user-queries
# Paste this JSON and hit Enter:
{"query_id":"test-1","user_id":"1","faculty":"CNTT","year":"2024","response_time_ms":100,"timestamp":"2025-12-08T10:00:00"}
# Ctrl+C to exit
```

### Bước 4: Check MongoDB for Analytics Data

```bash
# Run check script
python scripts/check_mongodb_data.py
```

**Expected after ~2-3 minutes**:

```
📊 Collection: query_analytics_by_faculty
   ✅ Found X documents

📊 Collection: query_analytics_by_year
   ✅ Found Y documents

📊 Collection: query_analytics_heatmap
   ✅ Found Z documents
```

---

## 🔍 Troubleshooting

### Kafka won't start:

```bash
# Check ports
netstat -ano | findstr "9092 2181"

# If ports in use, kill processes or change ports in docker-compose.kafka.yml
```

### Spark not processing:

```bash
# Check Spark logs for errors
docker logs spark-streaming --tail 100

# Common issues:
# - MongoDB connection failed → Check MONGO_URI in docker-compose.kafka.yml
# - Kafka topic not found → Will auto-create on first message
```

### No data in analytics collections after 5 minutes:

```bash
# 1. Check if queries are being published to Kafka
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic user-queries \
  --from-beginning \
  --max-messages 5

# Should see JSON query events

# 2. Check Spark logs for processing errors
docker logs spark-streaming | grep -i error
```

---

## 📊 Monitoring

### Kafka UI:

- **URL**: http://localhost:8080
- **Topic**: `user-queries`
- **Check**: Message count, consumer lag

### Spark UI:

- **URL**: http://localhost:4040 (only when job running)
- **Check**: Streaming batches, processing time

---

## 🛑 Stop Services

```bash
# Stop all
docker-compose -f docker-compose.kafka.yml stop

# Stop and remove
docker-compose -f docker-compose.kafka.yml down

# Stop and remove with volumes (CAUTION: deletes data)
docker-compose -f docker-compose.kafka.yml down -v
```

---

## ✅ Next Steps After Kafka/Spark Running:

1. **Make a few queries** in chat UI
2. **Wait 2-3 minutes** for Spark to process
3. **Check MongoDB** for analytics data
4. **Open Superset** (http://localhost:8089)
5. **Create datasets** from analytics collections
6. **Build dashboards** in Superset
7. **Update Dashboard IDs** in React AnalyticsView.jsx
8. **Test embedding** in React UI

Good luck! 🎉
