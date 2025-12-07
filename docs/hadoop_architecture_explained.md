# Kiến Trúc Hadoop - Giải Thích Chi Tiết

## Tại Sao Cần Hadoop?

Hadoop giải quyết 2 vấn đề chính của Big Data:

1. **Lưu trữ phân tán** (HDFS) - Lưu file lớn trên nhiều máy
2. **Xử lý phân tán** (YARN + MapReduce/Spark) - Xử lý data song song

## Các Thành Phần Chính

### 1. HDFS (Hadoop Distributed File System)

HDFS chia file thành các blocks và lưu trữ trên nhiều máy.

```
File 100GB → Chia thành 800 blocks (mỗi block 128MB)
              → Mỗi block được replicate 3 lần
              → Lưu trên các DataNodes khác nhau
```

#### NameNode (Master)

- **Vai trò**: Quản lý metadata của file system
- **Lưu gì**: Tên file, quyền, block nào nằm ở DataNode nào
- **Ví dụ**: Như "bảng mục lục" của thư viện
- **Số lượng**: 1 (hoặc 2 nếu có High Availability)

**Web UI**: http://localhost:9870

- Xem danh sách files
- Xem trạng thái DataNodes
- Browse HDFS như file explorer

#### DataNode (Workers)

- **Vai trò**: Lưu trữ data blocks thực tế
- **Lưu gì**: Các blocks của files
- **Ví dụ**: Như "kệ sách" chứa sách thật
- **Số lượng**: Ít nhất 1, thường 3-100+ trong production

**Web UI**: http://localhost:9864 (DataNode 1)

- Xem storage capacity
- Xem blocks đang lưu

---

### 2. YARN (Yet Another Resource Negotiator)

YARN quản lý tài nguyên (CPU, RAM) để chạy các jobs (MapReduce, Spark, etc.)

#### ResourceManager (Master)

- **Vai trò**: Phân bổ resources cho applications
- **Quản lý**: Memory, CPU cores cho các jobs
- **Ví dụ**: Như "người phân công công việc" cho nhân viên
- **Số lượng**: 1

**Web UI**: http://localhost:8088

- Xem applications đang chạy
- Monitor resource usage
- Job history

#### NodeManager (Workers)

- **Vai trò**: Chạy containers/tasks trên từng node
- **Quản lý**: Containers đang chạy, monitor resources
- **Ví dụ**: Như "nhân viên" thực hiện công việc được giao
- **Số lượng**: Ít nhất 1, thường bằng số DataNodes

**Web UI**: http://localhost:8042 (NodeManager 1)

- Xem containers đang chạy
- Monitor CPU/memory usage

---

## Tại Sao Cần Nhiều DataNodes & NodeManagers?

### Trong Production (Hệ Thống Thật)

#### 1. **Replication (Sao Lưu)**

```
File: document.pdf (128MB)
Block 1 → DataNode 1, DataNode 2, DataNode 3
         (3 copies để đảm bảo an toàn)

Nếu DataNode 2 hỏng → Vẫn có 2 copies khác!
```

**Mặc định**: Replication factor = 3

- 1 copy gốc
- 2 copies backup

#### 2. **Fault Tolerance (Chịu Lỗi)**

```
Scenario: 1 DataNode chết

❌ Chỉ 1 DataNode → Mất hết data, system down
✅ 3 DataNodes → 2 còn sống, system hoạt động bình thường
```

#### 3. **Parallel Processing (Xử Lý Song Song)**

```
Job: Đếm từ trong 1TB files

1 NodeManager → Chạy 1 máy → 10 giờ
3 NodeManagers → Chạy 3 máy song song → ~3.3 giờ
10 NodeManagers → Chạy 10 máy song song → ~1 giờ
```

#### 4. **Load Balancing (Cân Bằng Tải)**

```
1000 users đọc file cùng lúc

1 DataNode → Nghẽn, slow
3 DataNodes → Requests được phân tán, nhanh hơn
```

---

### Trong Testing (Môi Trường Test)

> **💡 Kết luận**: Với mục đích **CHỈ TEST**, bạn CHỈ CẦN:
>
> - **1 NameNode** (bắt buộc)
> - **1 DataNode** (đủ để lưu trữ)
> - **1 ResourceManager** (bắt buộc nếu dùng YARN)
> - **1 NodeManager** (đủ để chạy Spark jobs)

#### Ưu điểm của setup tối giản:

- ✅ **Ít RAM hơn**: ~2GB thay vì ~6GB
- ✅ **Start nhanh hơn**: 30s thay vì 60-90s
- ✅ **Đơn giản hơn**: Ít logs, dễ debug
- ✅ **Đủ để test**: Verify HDFS read/write, Spark jobs hoạt động

#### Nhược điểm:

- ❌ **Không test được replication**: Replication = 1 (không có backup)
- ❌ **Không test được fault tolerance**: 1 node chết = hệ thống chết
- ❌ **Không test được performance**: Không xử lý song song

#### Khuyến nghị:

```
Testing/Development  → 1 DataNode + 1 NodeManager (MINIMAL)
Staging/Pre-Prod    → 2-3 DataNodes + 2-3 NodeManagers
Production          → 5+ DataNodes + 5+ NodeManagers (FULL)
```

---

## So Sánh Cấu Hình

### Full Multi-Node Setup (Production-like)

```yaml
Services: 6
├── hadoop-namenode (1)
├── hadoop-datanode-1 (1)
├── hadoop-datanode-2 (1)
├── hadoop-resourcemanager (1)
├── hadoop-nodemanager-1 (1)
└── hadoop-nodemanager-2 (1)

Resources:
- RAM: ~6GB total
- Disk: ~3 volumes
- Startup: 60-90 seconds

Features:
✅ Replication factor 2
✅ Fault tolerance
✅ Parallel processing
✅ Gần giống production
```

### Minimal Setup (Testing Only)

```yaml
Services: 4
├── hadoop-namenode (1)
├── hadoop-datanode (1)         # CHỈ 1 THAY VÌ 2
├── hadoop-resourcemanager (1)
└── hadoop-nodemanager (1)      # CHỈ 1 THAY VÌ 2

Resources:
- RAM: ~2GB total
- Disk: ~2 volumes
- Startup: 30-40 seconds

Features:
✅ HDFS read/write works
✅ Spark jobs can run
⚠️ Replication factor 1 (no backup)
⚠️ No fault tolerance
⚠️ No parallel processing demo
```

---

## Khi Nào Dùng Gì?

### Dùng MINIMAL (1 DataNode + 1 NodeManager) khi:

- ✅ Bạn đang **học/test** Hadoop lần đầu
- ✅ Máy có **ít RAM** (< 8GB)
- ✅ Chỉ cần **verify HDFS operations** và **Spark jobs chạy được**
- ✅ **Không quan tâm** đến fault tolerance / replication

### Dùng FULL (2+ DataNodes + 2+ NodeManagers) khi:

- ✅ Test **replication** và **fault tolerance**
- ✅ Test **parallel processing** performance
- ✅ Chuẩn bị **deploy production**
- ✅ Demo cho **stakeholders** (giống production)

---

## Tóm Tắt

| Component           | Tối Thiểu | Khuyến Nghị Test | Production      |
| ------------------- | --------- | ---------------- | --------------- |
| **NameNode**        | 1         | 1                | 1 (+ 1 standby) |
| **DataNode**        | 1         | 2-3              | 5-100+          |
| **ResourceManager** | 1         | 1                | 1 (+ 1 standby) |
| **NodeManager**     | 1         | 2-3              | 5-100+          |

> **Cho mục đích TEST của bạn**: Dùng **MINIMAL** là đủ! Tôi sẽ tạo file `docker-compose.hadoop-minimal.yml` để bạn dễ dùng.

---

## Tài Liệu Tham Khảo

- [Hadoop HDFS Architecture](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
- [YARN Architecture](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html)
- [HDFS Replication](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html#Data_Replication)
