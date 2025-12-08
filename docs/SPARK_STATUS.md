## ⏳ Spark đang khởi tạo - Chờ download dependencies

### 📊 Tình trạng hiện tại:

✅ **Kafka** - Running, có messages  
⏳ **Spark** - Đang download JARs lần đầu  
❓ **MongoDB** - Chờ Spark process data

### 🔍 Spark logs đang show:

```
[SUCCESSFUL] org.apache.spark#spark-sql-kafka-0-10_2.12
[SUCCESSFUL] org.mongodb.spark#mongo-spark-connector_2.12
downloading hadoop-client-runtime...
```

→ Đây là **bình thường**! Spark cần download:

- Kafka connector (~4MB)
- MongoDB connector (~2MB)
- Hadoop libs (~15MB)

### ⏱️ Thời gian dự kiến:

- **Download**: 2-5 phút (tùy network)
- **Sau download**: Spark auto-start streaming job
- **Sau đó**: Data sẽ flow vào MongoDB trong vài phút

### 🎯 Các chờ Spark khởi động xong, bạn sẽ thấy:

```bash
docker logs spark-streaming --tail 20
```

**Output khi ready**:

```
INFO: Starting Spark Streaming Job...
INFO: Connecting to Kafka broker: kafka:29092
INFO: Faculty aggregation stream started
INFO: Year aggregation stream started
INFO: Heatmap aggregation stream started
INFO: All streaming queries started. Waiting for termination...
```

### 📝 Check lại sau 5 phút:

```bash
# 1. Verify Spark started
docker logs spark-streaming | findstr "stream started"

# 2. Check MongoDB data
python scripts/check_mongodb_data.py

# 3. Access Spark UI
# URL: http://localhost:4040
# (Only available after streaming job starts)
```

### ⚠️ Nếu download quá lâu (>10 phút):

```bash
# Restart Spark với clean cache
docker-compose -f docker-compose.kafka.yml restart spark-streaming

# Hoặc check network
docker logs spark-streaming | findstr "downloading"
```

---

**TL;DR**: Đợi thêm ~3-5 phút cho Spark download xong, rồi sẽ tự động start! 🚀
