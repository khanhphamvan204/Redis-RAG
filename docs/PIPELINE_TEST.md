# 🧪 Test Kafka-Spark-MongoDB Pipeline

## ✅ Spark đã start - Giờ test pipeline!

### 📝 Test Plan:

1. **Generate test queries** → FastAPI ghi log và publish Kafka
2. **Kafka receives** → Messages trong topic `user-queries`
3. **Spark consumes** → Real-time aggregation
4. **MongoDB receives** → Data trong analytics collections
5. **Verify** → Check MongoDB có data

---

## 🚀 Bước 1: Generate Test Queries

### Option A: Dùng Chat UI (Khuyến nghị)

```
1. Mở http://localhost:3000
2. Login
3. Hỏi 5-10 câu khác nhau
4. Mỗi câu sẽ tự động:
   - Log vào MongoDB `query_logs`
   - Publish lên Kafka `user-queries`
```

**Câu hỏi gợi ý**:

- "Thông tin về khoa CNTT"
- "Lịch học hôm nay"
- "Điểm thi cuối kỳ"
- "Giáo viên khoa Toán"
- "Quy định về học phí"

### Option B: Script Python (Nếu UI không dùng được)

```python
# test_query_generation.py
import requests

API_URL = "http://localhost:8000/documents/vector/search-with-llm-context"
TOKEN = "YOUR_JWT_TOKEN"  # Get from localStorage in browser

queries = [
    "Thông tin về khoa CNTT",
    "Lịch học hôm nay",
    "Điểm thi cuối kỳ",
    "Giáo viên khoa Toán",
    "Quy định về học phí"
]

headers = {"Authorization": f"Bearer {TOKEN}"}

for query in queries:
    payload = {
        "query": query,
        "file_type": "public",
        "k": 5,
        "session_id": "test_session_123"
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    print(f"✓ Query: {query} - Status: {response.status_code}")
```

---

## 🔍 Bước 2: Verify Kafka Received Messages

```bash
# Check topic message count (via Kafka UI)
# URL: http://localhost:8080
# Navigate to: Topics → user-queries → Messages

# OR check via CLI
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic user-queries \
  --from-beginning \
  --max-messages 5
```

**Expected**: JSON messages với query data

---

## ⚙️ Bước 3: Wait for Spark Processing

Spark processes data theo window:

- **Faculty aggregation**: 5-minute window, 1-minute slide
- **Year aggregation**: 5-minute window, 1-minute slide
- **Heatmap**: 15-minute window, 5-minute slide

**Đợi**: ~2-3 phút sau khi generate queries

---

## 📊 Bước 4: Check MongoDB Analytics Collections

### Run Check Script:

```bash
cd "E:\HK1 (2025 - 2026)\BigData\rag\redis_rag"
python scripts/check_mongodb_data.py
```

### Expected Output:

```
📊 Collection: query_logs
   ✅ Found 15 documents  # Raw queries

📊 Collection: query_analytics_by_faculty
   ✅ Found 3 documents   # Aggregated by faculty

   Sample document:
   {
     "window_start": "2025-12-08T11:00:00",
     "window_end": "2025-12-08T11:05:00",
     "faculty": "CNTT",
     "query_count": 5,
     "unique_users": 2,
     "avg_response_time": 1234.5,
     "avg_contexts_found": 4.2
   }

📊 Collection: query_analytics_by_year
   ✅ Found 2 documents   # Aggregated by year

📊 Collection: query_analytics_heatmap
   ✅ Found 4 documents   # Faculty x Year heatmap
```

---

## ✅ Success Criteria:

- [x] Chat UI hoạt động
- [x] Queries được log vào `query_logs`
- [x] Kafka topic `user-queries` có messages
- [x] Spark logs show "Batch X written to MongoDB"
- [x] Collections `query_analytics_*` có data

---

## 🐛 Troubleshooting:

### ❌ Queries không vào Kafka:

```bash
# Check FastAPI logs
# Tìm: "Query tracked" và "Publishing to Kafka"

# Nếu không thấy "Publishing to Kafka"
# → Check kafka_service.py import errors
```

### ❌ Spark không ghi MongoDB:

```bash
# Check Spark logs
docker logs spark-streaming | findstr "MongoDB"

# Common errors:
# - "Connection refused" → MongoDB không accessible từ container
# - "Authentication failed" → Sai credentials
```

### ❌ Collections vẫn rỗng after 5 minutes:

```bash
# 1. Check Spark có đang process không
docker logs spark-streaming | findstr "Batch"

# 2. Check window timing
# Spark chỉ flush data khi window kết thúc
# Nếu vừa generate queries, đợi thêm 5 phút

# 3. Manually trigger checkpoint
docker-compose -f docker-compose.kafka.yml restart spark-streaming
```

---

## 📈 Next Steps After Success:

1. ✅ Pipeline verified → Create Superset dashboards
2. ✅ Dashboards created → Get Dashboard IDs
3. ✅ Update IDs in `AnalyticsView.jsx`
4. ✅ Test embedding in React UI
5. ✅ Done! 🎉

---

**Good luck testing!** 🚀
