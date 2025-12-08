# Spark Streaming Setup Guide

Apache Spark Streaming cho real-time analytics từ Kafka.

## 🚀 Khởi động Spark Streaming

### Start toàn bộ stack (Kafka + Spark)

```bash
# Build và start tất cả services
docker-compose -f docker-compose.kafka.yml up -d --build

# Xem logs của Spark
docker logs spark-streaming -f

# Check status
docker ps | grep spark
```

### Stop services

```bash
docker-compose -f docker-compose.kafka.yml down

# Xóa cả volumes và checkpoints
docker-compose -f docker-compose.kafka.yml down -v
```

## 📊 Monitoring & UI

### 1. Spark UI

- **URL**: http://localhost:4040
- **Features**:
  - Streaming tab: Active streaming queries
  - SQL tab: Running aggregations
  - Jobs/Stages: Execution details
  - Executors: Resource usage

### 2. Kafka UI

- **URL**: http://localhost:8080
- **Monitor**:
  - Topic `user-queries` messages
  - Consumer group lag
  - Partition distribution

### 3. MongoDB

Connect để xem kết quả analytics:

```bash
mongosh "mongodb://admin:123@localhost:27017/faiss_db?authSource=admin"
```

Queries:

```javascript
// Xem analytics by faculty
db.query_analytics_by_faculty.find().sort({ window_start: -1 }).limit(5);

// Xem analytics by year
db.query_analytics_by_year.find().sort({ window_start: -1 }).limit(5);

// Xem heatmap data
db.query_analytics_heatmap.find().sort({ window_start: -1 }).limit(5);
```

## 🔍 Architecture Flow

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   FastAPI App   │
└────┬────────┬───┘
     │        │
     ▼        ▼
┌─────────┐  ┌──────────────┐
│ MongoDB │  │ Kafka Topic  │
│ (Logs)  │  │ user-queries │
└─────────┘  └──────┬───────┘
                    │
                    ▼
          ┌──────────────────┐
          │ Spark Streaming  │
          │ (3 Aggregations) │
          └────────┬─────────┘
                   │
                   ▼
          ┌─────────────────┐
          │    MongoDB      │
          │  (Analytics)    │
          └─────────────────┘
            • by_faculty
            • by_year
            • heatmap
```

## 📈 Streaming Aggregations

### 1. **By Faculty** (5-min window, 1-min slide)

Metrics:

- Query count
- Unique users
- Avg response time
- Avg contexts found
- Rewritten queries count
- History usage count

### 2. **By Year** (5-min window, 1-min slide)

Metrics:

- Query count per year
- Unique users
- Avg performance metrics

### 3. **Heatmap** (15-min window, 5-min slide)

Cross-analysis:

- Faculty × Year distribution
- Query patterns

## 🧪 Testing

### Test End-to-End Pipeline

```bash
# 1. Đảm bảo tất cả services đang chạy
docker ps

# Expected containers:
# - zookeeper
# - kafka
# - kafka-ui
# - spark-streaming
# - mongo-db (từ docker-compose.yml khác)

# 2. Submit test query qua FastAPI
# (Use Postman/curl hoặc qua UI)

# 3. Check Kafka message
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic user-queries \
  --from-beginning \
  --max-messages 1

# 4. Monitor Spark processing
docker logs spark-streaming --tail 50

# 5. Wait 5-15 phút (window time)

# 6. Query MongoDB cho analytics
mongosh "mongodb://admin:123@localhost:27017/faiss_db?authSource=admin"
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Kafka
KAFKA_BROKER_URL=localhost:9092
KAFKA_TOPIC_QUERIES=user-queries

# MongoDB (for analytics)
DATABASE_URL=mongodb://admin:123@host.docker.internal:27017/faiss_db?authSource=admin
```

### Spark Resources

Mặc định trong `docker-compose.kafka.yml`:

- Driver Memory: 2GB
- Executor Memory: 2GB
- Master: local[2] (2 cores)

Để thay đổi, edit `spark/entrypoint.sh`:

```bash
spark-submit \
    --conf spark.driver.memory=4g \
    --conf spark.executor.memory=4g \
    ...
```

## 🐛 Troubleshooting

### Spark container không start

**Check logs:**

```bash
docker logs spark-streaming
```

**Common issues:**

1. **Kafka not ready**: Spark đợi Kafka 30s, nếu không connect được sẽ fail

   - Solution: Đảm bảo Kafka healthy trước khi start Spark

2. **MongoDB connection error**:

   - Check MongoDB đang chạy: `docker ps | grep mongo`
   - Verify credentials trong MONGO_URI

3. **Port 4040 conflict**:
   - Change port trong `docker-compose.kafka.yml`: `"4041:4040"`

### No analytics data in MongoDB

**Possible reasons:**

1. **No query events**: Chưa có query nào qua hệ thống

   - Submit queries qua FastAPI

2. **Window chưa complete**: Aggregation windows cần thời gian (5-15 phút)

   - Đợi thêm và check lại

3. **Spark job error**: Check Spark logs
   ```bash
   docker logs spark-streaming | grep ERROR
   ```

### High memory usage

```bash
# Monitor resources
docker stats spark-streaming

# Reduce memory in entrypoint.sh
--conf spark.driver.memory=1g \
--conf spark.executor.memory=1g
```

## 📝 Logs Location

```bash
# Spark application logs
docker logs spark-streaming

# Spark checkpoints (persisted)
docker volume inspect redis_rag_spark-checkpoints
```

## 🔄 Restart Streaming Job

```bash
# Restart container
docker restart spark-streaming

# Rebuild nếu code thay đổi
docker-compose -f docker-compose.kafka.yml up -d --build spark-streaming

# Clear checkpoints để start fresh
docker-compose -f docker-compose.kafka.yml down -v
docker-compose -f docker-compose.kafka.yml up -d --build
```

## 📚 Related Documentation

- [Kafka Setup Guide](KAFKA_SETUP.md)
- [HDFS Setup Guide](HDFS_SETUP.md)
