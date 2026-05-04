# 🤖 Redis RAG — Hệ thống RAG Thông minh với Phân quyền theo Thư mục

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-7.0+-red.svg)](https://redis.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0+-brightgreen.svg)](https://mongodb.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange.svg)](https://ai.google.dev)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Redis RAG** là hệ thống **Retrieval-Augmented Generation** được xây dựng bằng FastAPI, sử dụng **Redis Vector Search** để tìm kiếm ngữ nghĩa và tích hợp **phân quyền truy cập theo phòng/khoa (department-based access control)** thông qua JWT Authentication. Người dùng chỉ có thể truy vấn và nhận kết quả từ đúng những thư mục tài liệu mà họ được cấp quyền.

---

## 📐 Kiến trúc hệ thống

Hệ thống gồm **hai luồng chính**:

<p align="center">
  <img src="docs/rag_ingestion_flow.png" width="48%" alt="Document Ingestion Flow" />
  &nbsp;&nbsp;
  <img src="docs/rag_query_flow.png" width="48%" alt="RAG Query Flow" />
</p>
<p align="center">
  <em>Trái: Document Ingestion Flow &nbsp;|&nbsp; Phải: RAG Query Flow với Department-based Access Control</em>
</p>


### 1️⃣ Document Ingestion Flow — Luồng nạp tài liệu

<p align="center">
  <img src="docs/rag_ingestion_flow.png" width="70%" alt="Document Ingestion Flow" />
</p>

```
User Upload
    │
    ▼
[1] FastAPI — Xác thực JWT & Extract department_id
    │
    ▼
[2] Lưu file vào thư mục theo phòng/khoa
    Cấu trúc: Root_Folder/{dept_id}/File_Folder/
    │
    ▼
[3] Extract & Chunk Text (PDF, DOCX, TXT, CSV, XLSX)
    │
    ▼
[4] Google Gemini — Generate Embeddings (vector 768 chiều)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
[5a] Redis — Lưu Vectors          [5b] MongoDB — Lưu Metadata
     (với tag department_id)            (file_name, dept, chunk_info...)
```

**Mỗi vector được gắn tag `department_id`** khi lưu vào Redis, ví dụ:
```json
{
  "vector": [...],
  "department_id": "CS_dept",
  "doc_id": "uuid-xxx",
  "chunk_index": 2,
  "content": "Nội dung đoạn văn..."
}
```

---

### 2️⃣ RAG Query Flow — Luồng truy vấn

<p align="center">
  <img src="docs/rag_query_flow.png" width="80%" alt="RAG Query Flow" />
</p>

```
User gửi câu hỏi
    │
    ▼
[1] FastAPI — Xác thực JWT → Extract department_id = "CS_dept"
    │
    ▼
[2] Google Gemini — Tạo Query Embedding
    │
    ▼
[3] Redis Vector Search — Filter THEO department_id
     Query: @department_id:{CS_dept}
    │
    ▼
[4] Redis trả về chỉ các docs KHỚP với dept của user
    │
    ▼
[5] Build Context từ các chunks phù hợp
    │
    ▼
[6] Google Gemini — Generate Answer
    │
    ▼
[7] Trả kết quả về cho User
```

---

## 🔐 Phân quyền theo Thư mục (Department-based Access Control)

Đây là tính năng **cốt lõi** của hệ thống. Mỗi user khi đăng nhập sẽ nhận JWT token chứa thông tin `department_id`. Khi truy vấn RAG, hệ thống **tự động lọc** kết quả theo đúng phòng/khoa của user đó.

```
┌─────────────────────────────────────────────────────────────┐
│              Department-based Access Control                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User 1 (CS_dept)   ──►  📄 CS docs only      ✅          │
│                            📄 Math docs         ❌          │
│                            📄 Physics docs      ❌          │
│                                                             │
│   User 2 (Math_dept) ──►  📄 CS docs           ❌          │
│                            📄 Math docs only    ✅          │
│                            📄 Physics docs      ❌          │
│                                                             │
│   User 3 (Physics)   ──►  📄 CS docs           ❌          │
│                            📄 Math docs         ❌          │
│                            📄 Physics docs only ✅          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Cơ chế hoạt động

| Bước | Thành phần | Mô tả |
|------|-----------|-------|
| **Auth** | FastAPI + JWT | Decode token → lấy `department_id` |
| **Upload** | FastAPI | Lưu file vào `Root_Folder/{dept_id}/File_Folder/` |
| **Index** | Redis Vector | Tag vector với `department_id` |
| **Query** | Redis Filter | `@department_id:{dept_id}` — chỉ tìm trong dept của user |
| **Result** | FastAPI | Trả về answer được tạo từ đúng context của dept |

---

## 🗂️ Cấu trúc thư mục tài liệu

Tài liệu được tổ chức theo phòng/khoa, mỗi thư mục là một **vùng dữ liệu riêng biệt** với phân quyền độc lập:

```
Root_Folder/
├── 🌐 Public_Rag_Info/          # Tài liệu công khai — mọi user đều đọc được
│   └── File_Folder/             # Lưu files gốc
│
├── 🎓 TaiLieuMonHoc_CNTT/       # Tài liệu Khoa CNTT
│   └── File_Folder/
│
├── 📘 DeAnTotNghiep/            # Tài liệu đề án tốt nghiệp
│   └── File_Folder/
│
└── ⚙️ {department_id}/          # Bất kỳ phòng/khoa nào thêm qua API
    └── File_Folder/
```

> **Lưu ý:** Danh sách thư mục (folders) được quản lý **động** qua MongoDB. Admin có thể thêm/xóa thư mục mà không cần sửa code, hệ thống tự động nhận diện và cấu hình lại.

---

## 🧰 Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|-----------|----------|---------|
| **API Server** | FastAPI | REST API, JWT Auth, routing |
| **Vector DB** | Redis (RedisSearch) | Lưu & tìm kiếm vector theo dept filter |
| **Metadata DB** | MongoDB | Lưu thông tin tài liệu, chunk metadata |
| **Embedding** | Google Gemini API | Tạo vector từ text (768 chiều) |
| **LLM** | Google Gemini | Sinh câu trả lời từ context |
| **Auth** | JWT (PyJWT) | Xác thực user và extract department_id |
| **File Parser** | PyMuPDF, python-docx, pandas | Đọc PDF, DOCX, TXT, CSV, XLSX |

---

## ⚡ Cài đặt và chạy

### Yêu cầu hệ thống

- **Python**: 3.10+
- **Redis**: 7.0+ với module `RedisSearch` (RedisStack hoặc Redis Enterprise)
- **MongoDB**: 6.0+
- **RAM**: 4GB+ (khuyến nghị)
- **OS**: Windows, macOS, Linux

---

### 1. Clone repository

```bash
git clone https://github.com/khanhphamvan204/Redis-RAG.git
cd Redis-RAG
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

Tạo file `.env` từ template:

```bash
cp .env.example .env
```

Chỉnh sửa `.env`:

```env
# ============================================
# CORE CONFIGURATION
# ============================================

# Google Gemini — dùng cho Embedding & LLM
GOOGLE_API_KEY=your_google_api_key_here

# JWT Secret — dùng để ký và verify token
JWT_SECRET_KEY=your_jwt_secret_key_here

# ============================================
# DATABASE
# ============================================

# MongoDB — lưu metadata tài liệu & cấu hình folders
DATABASE_URL=mongodb://admin:123@localhost:27017

# Redis — Vector database với RedisSearch
REDIS_URL=redis://localhost:6379

# ============================================
# STORAGE
# ============================================

# Thư mục gốc chứa tài liệu theo dept
DATA_PATH=Root_Folder
```

---

### 5. Khởi động services

**Khởi động Redis (với RedisSearch):**
```bash
# Docker (khuyến nghị)
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```

**Khởi động MongoDB:**
```bash
# Ubuntu/Debian
sudo systemctl start mongod

# macOS với Homebrew
brew services start mongodb-community

# Windows
net start MongoDB
```

**Khởi động API:**
```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
python main.py
```

API khởi chạy tại: `http://localhost:8000`  
Swagger Docs: `http://localhost:8000/docs`

---

## 🐳 Docker Compose

Chạy toàn bộ hệ thống (FastAPI + Redis + MongoDB) với một lệnh:

```bash
docker-compose up -d --build
```

### `docker-compose.yml`

```yaml
version: "3.8"

services:
  # FastAPI Application
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: redis-rag-api
    ports:
      - "8000:8000"
    volumes:
      - ./Root_Folder:/app/Root_Folder
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1
      - DATABASE_URL=mongodb://admin:123@mongo:27017/faiss_db?authSource=admin
      - REDIS_URL=redis://redis:6379
    networks:
      - rag-network
    depends_on:
      mongo:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  # Redis Vector Database
  redis:
    image: redis/redis-stack-server:latest
    container_name: redis-rag-vector
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rag-network
    restart: unless-stopped

  # MongoDB Metadata Database
  mongo:
    image: mongo:6.0
    container_name: redis-rag-mongo
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
      - ./mongo-init:/docker-entrypoint-initdb.d
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=123
      - MONGO_INITDB_DATABASE=faiss_db
    healthcheck:
      test: |
        mongosh --host localhost --port 27017 \
                --username admin --password 123 \
                --authenticationDatabase admin \
                --eval "db.adminCommand('ping')"
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s
    networks:
      - rag-network
    restart: unless-stopped

volumes:
  redis-data:
    driver: local
  mongo-data:
    driver: local

networks:
  rag-network:
    driver: bridge
```

### Các lệnh Docker thông dụng

```bash
# Khởi động tất cả services
docker-compose up -d --build

# Xem logs
docker-compose logs -f
docker-compose logs -f app
docker-compose logs -f redis

# Kiểm tra health
docker ps
curl http://localhost:8000/health

# Dừng và xóa
docker-compose down
docker-compose down -v   # ⚠️ xóa cả data volumes
```

---

## 📋 API Endpoints

### Authentication

| Method | Endpoint | Mô tả |
|--------|---------|-------|
| `POST` | `/auth/login` | Đăng nhập, nhận JWT token |
| `POST` | `/auth/register` | Đăng ký tài khoản |

> **JWT Payload** cần có trường `department_id` để hệ thống phân quyền đúng:
> ```json
> {
>   "sub": "user_001",
>   "department_id": "CS_dept",
>   "role": "teacher"
> }
> ```

---

### Quản lý thư mục (Folders)

| Method | Endpoint | Mô tả |
|--------|---------|-------|
| `GET` | `/folders` | Lấy danh sách thư mục |
| `POST` | `/folders` | Tạo thư mục mới (Admin) |
| `DELETE` | `/folders/{folder_name}` | Xóa thư mục (Admin) |

---

### Quản lý tài liệu

#### Upload tài liệu

```
POST /documents/vector/add
Authorization: Bearer <JWT_TOKEN>
Content-Type: multipart/form-data
```

| Tham số | Kiểu | Bắt buộc | Mô tả |
|---------|------|---------|-------|
| `file` | File | ✅ | File cần upload (PDF, DOCX, TXT, CSV, XLSX) |
| `uploaded_by` | string | ✅ | Tên/ID người upload |
| `file_type` | string | ✅ | Tên thư mục/dept đích |

> Hệ thống tự động extract `department_id` từ JWT token để gắn tag vào vector.

#### Lấy danh sách tài liệu

```
GET /documents/list?file_type={dept}&limit={n}&skip={m}
Authorization: Bearer <JWT_TOKEN>
```

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `file_type` | Lọc theo thư mục/dept | tất cả |
| `limit` | Số lượng kết quả | 100 |
| `skip` | Bỏ qua N bản đầu | 0 |

#### Xóa tài liệu

```
DELETE /documents/vector/{doc_id}
Authorization: Bearer <JWT_TOKEN>
```

---

### RAG Query

```
POST /query
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json

{
  "question": "Hãy giải thích về thuật toán sắp xếp nhanh?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Thuật toán QuickSort hoạt động bằng cách...",
  "sources": [
    {
      "doc_id": "uuid-xxx",
      "file_name": "giao_trinh_cau_truc_du_lieu.pdf",
      "chunk_index": 3,
      "score": 0.92,
      "department_id": "CS_dept"
    }
  ],
  "department_id": "CS_dept"
}
```

> Hệ thống **chỉ tìm kiếm trong thư mục dept của user** — user thuộc `CS_dept` sẽ không bao giờ nhận được kết quả từ `Math_dept`.

---

## 🔧 Sử dụng API

### Python Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Đăng nhập lấy token
login_resp = requests.post(f"{BASE_URL}/auth/login", json={
    "username": "nguyen_van_a",
    "password": "secret"
})
token = login_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload tài liệu (tự động gắn tag dept từ JWT)
with open("giao_trinh.pdf", "rb") as f:
    upload_resp = requests.post(
        f"{BASE_URL}/documents/vector/add",
        headers=headers,
        files={"file": f},
        data={
            "uploaded_by": "nguyen_van_a",
            "file_type": "TaiLieuMonHoc_CNTT"
        }
    )
print(upload_resp.json())

# 3. Truy vấn RAG (tự động lọc theo dept)
query_resp = requests.post(f"{BASE_URL}/query", headers=headers, json={
    "question": "Giải thích về độ phức tạp thuật toán O(n log n)?",
    "top_k": 5
})
print(query_resp.json()["answer"])
```

### cURL Examples

```bash
# Đăng nhập
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"nguyen_van_a","password":"secret"}'

# Upload tài liệu
curl -X POST http://localhost:8000/documents/vector/add \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@giao_trinh.pdf" \
  -F "uploaded_by=nguyen_van_a" \
  -F "file_type=TaiLieuMonHoc_CNTT"

# Truy vấn RAG
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"question":"Giải thích về cấu trúc dữ liệu stack?","top_k":5}'

# Health check
curl http://localhost:8000/health
```

---

## ⚙️ Biến môi trường

| Biến | Mô tả | Bắt buộc |
|------|-------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key (Embedding + LLM) | ✅ |
| `JWT_SECRET_KEY` | Secret để ký JWT token | ✅ |
| `DATABASE_URL` | MongoDB connection string | ✅ |
| `REDIS_URL` | Redis connection string | ✅ |
| `DATA_PATH` | Thư mục gốc lưu tài liệu | `Root_Folder` |

---

## 🔍 Luồng phân quyền chi tiết

```
              ┌──────────────────────────────────────────┐
              │          JWT Token Payload               │
              │  {                                       │
              │    "sub": "user_001",                    │
              │    "department_id": "CS_dept",  ◄────────┼── Trích xuất tại mọi request
              │    "role": "teacher"                     │
              │  }                                       │
              └──────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
           [Upload Document]            [Query RAG]
                    │                           │
                    ▼                           ▼
        Lưu vào:                    Redis Filter:
        Root_Folder/                @department_id:{CS_dept}
        CS_dept/File_Folder/                    │
                    │                           ▼
                    ▼               Chỉ trả về docs thuộc
        Redis Vector:               CS_dept — KHÔNG LỌT
        department_id = CS_dept     sang dept khác
```

---

## 📊 Hiệu suất

| Chỉ số | Giá trị |
|--------|---------|
| Upload & index tài liệu | ~2–5 giây / trang PDF |
| Vector search với filter | < 50ms cho 100K documents |
| Sinh câu trả lời (Gemini) | ~2–5 giây |
| Concurrent requests | 50+ (với uvicorn workers) |
| RAM sử dụng | ~512MB – 1GB (app) |

---

## 🐛 Troubleshooting

### Redis không kết nối được

```bash
# Kiểm tra Redis đang chạy
docker ps | grep redis

# Test kết nối
redis-cli -u redis://localhost:6379 ping
# Expected: PONG

# Kiểm tra RedisSearch module đã load chưa
redis-cli MODULE LIST
```

### MongoDB không kết nối được

```bash
# Kiểm tra MongoDB
sudo systemctl status mongod

# Test từ Docker container
docker-compose exec app python -c "
from pymongo import MongoClient
client = MongoClient('mongodb://admin:123@mongo:27017/?authSource=admin')
print('OK:', client.server_info()['version'])
"
```

### JWT token không hợp lệ

- Đảm bảo `JWT_SECRET_KEY` trong `.env` khớp với key dùng để ký token
- Kiểm tra token chưa hết hạn (`exp` claim)
- Đảm bảo payload có trường `department_id`

### Thư mục dept không tồn tại

```bash
# Tạo thư mục thủ công
mkdir -p Root_Folder/CS_dept/File_Folder

# Hoặc thêm qua API
curl -X POST http://localhost:8000/folders \
  -H "Authorization: Bearer <ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"folder_name": "CS_dept"}'
```

---

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## 🤝 Contributing

1. Fork project
2. Tạo feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'feat: add your feature'`
4. Push: `git push origin feature/your-feature`
5. Mở Pull Request

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) — Web framework hiện đại, hiệu suất cao
- [Redis / RedisStack](https://redis.io/docs/stack/) — Vector database với filter theo metadata
- [Google Gemini](https://ai.google.dev/) — Embedding & Language Model
- [MongoDB](https://mongodb.com/) — Lưu trữ metadata linh hoạt
- [LangChain](https://langchain.com/) — LLM orchestration framework
