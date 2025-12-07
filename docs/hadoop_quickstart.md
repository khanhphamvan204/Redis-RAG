# Hadoop Quick Start Guide - Testing Version

## Hướng Dẫn Nhanh Cho Testing

Đây là hướng dẫn sử dụng **version tối giản** của Hadoop chỉ để test.

### Bước 1: Tạo Network (nếu chưa có)

```bash
docker network create app-network
```

### Bước 2: Khởi Động Hadoop (Version Tối Giản)

```bash
# Từ thư mục redis_rag
docker-compose -f docker-compose.hadoop.yml up -d
```

⏱️ **Thời gian chờ**: ~30-40 giây để các services khởi động

### Bước 3: Kiểm Tra Services

```bash
# Xem tất cả containers
docker ps --filter "name=hadoop"

# Kiểm tra logs
docker logs hadoop-namenode
docker logs hadoop-datanode
```

**Expected Output:**

```
CONTAINER ID   IMAGE                                              STATUS
abc123...      bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8   Up (healthy)
def456...      bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8   Up (healthy)
ghi789...      bde2020/hadoop-resourcemanager:...                Up (healthy)
jkl012...      bde2020/hadoop-nodemanager:...                    Up (healthy)
```

### Bước 4: Mở Web UIs

| Service             | URL                   | Mô Tả                                  |
| ------------------- | --------------------- | -------------------------------------- |
| **NameNode**        | http://localhost:9870 | Browse HDFS files, xem DataNode status |
| **DataNode**        | http://localhost:9864 | Xem storage, blocks                    |
| **ResourceManager** | http://localhost:8088 | YARN cluster, running apps             |
| **NodeManager**     | http://localhost:8042 | Container status                       |

### Bước 5: Test HDFS Basic Operations

#### Cách 1: Dùng CLI (trong container)

```bash
# Vào NameNode container
docker exec -it hadoop-namenode bash

# Tạo thư mục
hdfs dfs -mkdir -p /user/root
hdfs dfs -mkdir -p /data/test

# Upload file
echo "Hello HDFS!" > /tmp/test.txt
hdfs dfs -put /tmp/test.txt /user/root/

# List files
hdfs dfs -ls /user/root/

# Read file
hdfs dfs -cat /user/root/test.txt

# Delete file
hdfs dfs -rm /user/root/test.txt

# Check HDFS status
hdfs dfsadmin -report
```

#### Cách 2: Dùng Python

```bash
# Vào app container
docker exec -it redis-rag bash

# Run test script
python test_hadoop_integration.py
```

**Expected Output:**

```
╔══════════════════════════════════════════════════════════════════╗
║               HADOOP HDFS INTEGRATION TEST SUITE                 ║
╚══════════════════════════════════════════════════════════════════╝

=================================================================
  Testing HDFS Basic Operations
=================================================================

1. Creating test directory...
✓ Directory created successfully

2. Writing test file...
✓ File written

3. Reading test file...
✓ File read successfully

...

╔══════════════════════════════════════════════════════════════════╗
║                         TEST SUMMARY                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Tests: 5     Passed: 5     Failed: 0                      ║
╚══════════════════════════════════════════════════════════════════╝

🎉 All tests passed successfully! Hadoop HDFS is ready for Phase 3.
```

### Bước 6: Dừng Hadoop

```bash
# Stop tất cả services
docker-compose -f docker-compose.hadoop-minimal.yml down

# Stop và xóa volumes (xóa hết data)
docker-compose -f docker-compose.hadoop-minimal.yml down -v
```

---

## Sử Dụng HDFS Trong Code

### Import Service

```python
from app.services.hdfs_service import hdfs_service
```

### Upload File

```python
# Upload file từ local vào HDFS
hdfs_service.upload_file(
    local_path="/app/Root_Folder/document.pdf",
    hdfs_path="/data/documents/document.pdf",
    overwrite=True
)
```

### Write Content

```python
# Write string content trực tiếp
import json

data = {"query": "test", "faculty": "CS"}
hdfs_service.write_file(
    hdfs_path="/data/logs/query.json",
    content=json.dumps(data),
    overwrite=True
)
```

### Read File

```python
# Read file content
content = hdfs_service.read_file("/data/logs/query.json")
print(content)  # {"query": "test", "faculty": "CS"}
```

### List Files

```python
# List files trong directory
files = hdfs_service.list_files("/data/logs")
print(files)  # ['query.json', 'query2.json', ...]
```

### Check if Exists

```python
# Check file/directory exists
exists = hdfs_service.exists("/data/logs/query.json")
print(exists)  # True
```

### Delete

```python
# Delete file
hdfs_service.delete("/data/logs/query.json")

# Delete directory recursively
hdfs_service.delete("/data/logs", recursive=True)
```

---

## Use Case Examples

### 1. Lưu Query Logs

```python
from datetime import datetime
import json
from app.services.hdfs_service import hdfs_service

def log_query(query: str, faculty: str, year: int):
    # Partition by date and hour
    now = datetime.now()
    log_dir = f"/data/query_logs/date={now.strftime('%Y-%m-%d')}/hour={now.hour:02d}"

    # Create directory
    hdfs_service.create_directory(log_dir)

    # Write log
    log_entry = {
        "timestamp": now.isoformat(),
        "query": query,
        "faculty": faculty,
        "year": year
    }

    filename = f"query_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    hdfs_service.write_file(
        hdfs_path=f"{log_dir}/{filename}",
        content=json.dumps(log_entry)
    )
```

### 2. Lưu Documents

```python
def store_document(doc_id: str, content: str, metadata: dict):
    # Partition by date
    now = datetime.now()
    doc_dir = f"/data/documents/{now.strftime('%Y/%m/%d')}"

    hdfs_service.create_directory(doc_dir)

    # Store document
    hdfs_service.write_file(
        hdfs_path=f"{doc_dir}/{doc_id}.txt",
        content=content,
        overwrite=True
    )

    # Store metadata
    hdfs_service.write_file(
        hdfs_path=f"{doc_dir}/{doc_id}_meta.json",
        content=json.dumps(metadata),
        overwrite=True
    )
```

---

## Troubleshooting

### Issue: Container "unhealthy"

```bash
# Check logs
docker logs hadoop-namenode
docker logs hadoop-datanode

# Restart
docker-compose -f docker-compose.hadoop-minimal.yml restart
```

### Issue: "Connection refused" từ Python

```bash
# Check namenode hostname resolution
docker exec -it redis-rag ping hadoop-namenode

# Check port
docker exec -it redis-rag curl http://hadoop-namenode:9870
```

### Issue: "No space left"

```bash
# Check volume usage
docker exec -it hadoop-namenode df -h

# Clear old data
docker exec -it hadoop-namenode hdfs dfs -rm -r /data/old_logs
```

---

## So Sánh: Minimal vs Full Setup

| Aspect                  | Minimal (Testing) | Full (Production-like) |
| ----------------------- | ----------------- | ---------------------- |
| **DataNodes**           | 1                 | 2                      |
| **NodeManagers**        | 1                 | 2                      |
| **RAM Usage**           | ~2GB              | ~6GB                   |
| **Startup Time**        | 30-40s            | 60-90s                 |
| **Replication**         | 1 (no backup)     | 2 (có backup)          |
| **Fault Tolerance**     | ❌                | ✅                     |
| **Parallel Processing** | ❌                | ✅                     |
| **Good For**            | Learning, Testing | Staging, Demo          |

**Khuyến nghị**: Dùng **minimal** cho testing, chuyển sang **full** khi cần demo hoặc sắp deploy.

---

## Chuyển Sang Full Setup

Khi bạn đã quen và muốn test replication / fault tolerance:

```bash
# Stop minimal setup
docker-compose -f docker-compose.hadoop-minimal.yml down

# Start full setup
docker-compose -f docker-compose.hadoop.yml up -d
```

**Lưu ý**: Data sẽ mất khi chuyển đổi. Backup nếu cần.
