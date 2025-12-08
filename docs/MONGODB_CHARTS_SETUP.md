# Metabase Setup Guide

Hướng dẫn setup Metabase để visualize real-time analytics từ Spark Streaming.

## 📋 Tổng quan

Metabase là open-source BI tool, hỗ trợ:

- ✅ **Native MongoDB support** - plugin built-in
- ✅ **Real-time data** từ MongoDB collections
- ✅ **Dễ setup** với Docker - 1 command
- ✅ **Beautiful UI** - user-friendly
- ✅ **Embedding** vào web apps
- ✅ **Auto-refresh** dashboards

**Ưu điểm so với MongoDB Charts**:

- ✅ Không có Docker compatibility issues
- ✅ Setup nhanh hơn (5 phút vs 15 phút)
- ✅ Community lớn, nhiều tutorials
- ✅ Có thể connect nhiều databases khác

## 🚀 Quick Start

### 1. Start Metabase Container

```bash
cd redis_rag
docker-compose -f docker-compose.charts.yml up -d
```

**Wait ~30 seconds** để Metabase khởi động lần đầu.

Kiểm tra container:

```bash
docker ps | grep metabase
docker logs metabase-analytics
```

### 2. Access Metabase UI

Mở browser và truy cập:

```
http://localhost:8090
```

**Lần đầu tiên**: Bạn sẽ thấy welcome screen.

## 🔧 Initial Setup Wizard

### Step 1: Language Selection

- Chọn **English** hoặc ngôn ngữ bạn muốn
- Click **Let's get started**

### Step 2: Create Admin Account

- **First name**: Admin
- **Last name**: User
- **Email**: admin@example.com (hoặc email của bạn)
- **Password**: Chọn password mạnh (ví dụ: `admin123`)
- Click **Next**

### Step 3: Add Database

1. **Database type**: Select **MongoDB** từ dropdown

2. **Display name**: `Local MongoDB`

3. **Host**: `host.docker.internal`

4. **Port**: `27017`

5. **Database name**: `faiss_db`

6. **Username**: Để trống (MongoDB không có auth)

7. **Password**: Để trống

8. **Additional Options** (expand):
   - **Use a secure connection (SSL)**: ❌ OFF
   - **Authenticate database name**: Để trống
9. Click **Connect database**

> ⚠️ Nếu connection failed, thử dùng IP thay vì `host.docker.internal`:
>
> ```bash
> ipconfig  # Get your local IP
> # Use IP như: 192.168.1.100
> ```

### Step 4: Data Preferences

- **Allow Metabase to anonymously collect usage events**: Tùy chọn (recommended: OFF cho privacy)
- Click **Next**

### Step 5: Complete Setup

- Click **Take me to Metabase**

🎉 **Setup hoàn tất!** Bạn sẽ thấy Metabase home page.

## 📊 Creating Analytics Questions (Charts)

Trong Metabase, charts được gọi là **Questions**.

### Question 1: Queries by Faculty (Bar Chart)

1. Click **New** → **Question**

2. **Select data**:

   - Database: `Local MongoDB`
   - Collection: `query_analytics_by_faculty`

3. **Summarize**:

   - Click **Summarize**
   - Metric: Count of rows (hoặc Sum of `query_count`)
   - Group by: `faculty`

4. **Visualization**:

   - Click **Visualization** button (chart icon)
   - Select **Bar Chart**
   - X-axis: `faculty`
   - Y-axis: Count

5. **Settings** (gear icon):

   - **Title**: "Queries by Faculty"
   - **X-axis label**: "Faculty"
   - **Y-axis label**: "Number of Queries"
   - **Color**: Choose color

6. **Save**:
   - Click **Save** (top right)
   - Name: "Queries by Faculty"
   - Description: "Total queries grouped by faculty"
   - **Create a new dashboard**: "Query Analytics Dashboard"
   - Click **Save**

### Question 2: Query Trends by Year (Line Chart)

1. Click **New** → **Question**

2. **Select data**:

   - Database: `Local MongoDB`
   - Collection: `query_analytics_by_year`

3. **Summarize**:

   - Metric: Sum of `query_count`
   - Group by: `year`

4. **Visualization**:

   - Select **Line Chart**
   - X-axis: `year`
   - Y-axis: Sum of query_count
   - **Sort**: By year ascending

5. **Settings**:

   - Title: "Query Trends by Year"
   - Enable **Show values on data points**

6. **Save**:
   - Name: "Query Trends by Year"
   - Add to dashboard: **Query Analytics Dashboard**
   - Click **Save**

### Question 3: Usage Heatmap

1. Click **New** → **Question**

2. **Select data**:

   - Collection: `query_analytics_heatmap`

3. **Summarize**:

   - Metric: Sum of `query_count`
   - Group by: `day_of_week`, then `hour`

4. **Visualization**:

   - Select **Table** (Metabase không có heatmap native, dùng table hoặc pivot)
   - Hoặc **Pivot Table**:
     - Rows: `day_of_week`
     - Columns: `hour`
     - Values: Sum of `query_count`

5. **Settings**:

   - Title: "Usage Heatmap (Hour × Day)"
   - **Conditional formatting**: Add color scale based on values

6. **Save**:
   - Name: "Usage Heatmap"
   - Add to dashboard: **Query Analytics Dashboard**

### Additional Question: Total Queries Count

1. **New Question** → Collection: `query_analytics_by_faculty`
2. **Summarize**: Sum of `query_count` (no grouping)
3. **Visualization**: **Number** (big single number)
4. **Settings**:
   - Title: "Total Queries"
   - **Number formatting**: Add thousands separator
5. **Save** to dashboard

## 📱 Dashboard Configuration

### 1. Navigate to Dashboard

- Click **Dashboards** (top menu)
- Select **Query Analytics Dashboard**

### 2. Arrange Charts

- Click **Edit dashboard** (pencil icon)
- **Drag and drop** questions to arrange
- **Resize** by dragging corners
- Suggested layout:
  ```
  [Total Queries]     [Queries by Faculty]
  [Query Trends]      [Usage Heatmap]
  ```
- Click **Save**

### 3. Enable Auto-Refresh

1. Click **dashboard settings** (gear icon)
2. **Auto-refresh**: Select **1 minute** (hoặc 5 minutes)
3. **Cache TTL**: Set to 60 seconds
4. Click **Done**

### 4. Enable Public Sharing (for Embedding)

1. Click **sharing icon** (share button)
2. **Enable sharing**:
   - Toggle **Public link** ON
   - Copy the **Public link**
3. **Embedding**:
   - Toggle **Enable embedding** ON
   - Copy **Embed code** hoặc **iframe URL**

Sample iframe URL format:

```
http://localhost:8090/public/dashboard/<HASH>
```

### 5. Get Individual Question Embed URLs

Cho mỗi question trong dashboard:

1. Open question
2. Click **sharing icon**
3. Enable **Public link**
4. Copy public URL format: `http://localhost:8090/public/question/<HASH>`

## 🔗 Update Backend with Embed URLs

Edit file: `redis_rag/app/api/routes/analytics.py`

Update trong hàm `get_charts_embed_info()`:

```python
"embed_urls": {
    "overview_dashboard": "http://localhost:8090/public/dashboard/<YOUR_DASHBOARD_HASH>?bordered=false&titled=false",
    "faculty_chart": "http://localhost:8090/public/question/<FACULTY_QUESTION_HASH>?bordered=false&titled=false",
    "year_chart": "http://localhost:8090/public/question/<YEAR_QUESTION_HASH>?bordered=false&titled=false",
    "heatmap_chart": "http://localhost:8090/public/question/<HEATMAP_QUESTION_HASH>?bordered=false&titled=false"
}
```

**URL Parameters** (optional):

- `?bordered=false` - Remove border
- `?titled=false` - Hide title (already shown in React)
- `?theme=night` - Dark mode
- `?refresh=60` - Auto-refresh interval (seconds)

## ✅ Verification

### 1. Test Questions in Metabase

- Questions hiển thị data từ MongoDB
- Charts render đúng
- Auto-refresh works (edit question → data updates)

### 2. Test Public Sharing

- Open public URL trong incognito browser
- Chart/Dashboard hiển thị không cần login

### 3. Test Backend API

```bash
# Check health
curl http://localhost:8000/analytics/health

# Get embed info
curl http://localhost:8000/analytics/charts/embed-info

# Get collection stats
curl http://localhost:8000/analytics/collections/stats
```

### 4. Test Frontend

1. **Update** `.env` (nếu cần):

   ```env
   MONGODB_CHARTS_URL=http://localhost:8090
   ```

2. Start frontend: `npm run dev`

3. Login vào app

4. Click tab **Analytics**

5. Charts hiển thị (sau khi config embed URLs)

6. Auto-refresh works

### 5. Test Real-time Updates

1. Tạo vài queries trong Chat view
2. Spark process và update MongoDB (~10-20 seconds)
3. Wait for Metabase cache to expire (~60 seconds)
4. Dashboard auto-refreshes và hiển thị data mới

## 🛠 Troubleshooting

### Cannot connect to MongoDB

**Error**: "Unable to connect to database"

**Solutions**:

1. Check MongoDB running: `mongosh --host localhost --port 27017`
2. Verify Metabase container can reach host:
   ```bash
   docker exec -it metabase-analytics ping host.docker.internal
   ```
3. Try using IP instead of `host.docker.internal`:
   ```bash
   ipconfig  # Windows
   # Use your local IP: 192.168.x.x
   ```
4. Check firewall không block port 27017

### Collections not showing data

**Error**: "No results"

**Solutions**:

1. Verify Spark đang chạy: `docker ps | grep spark`
2. Check MongoDB có data:
   ```bash
   mongosh
   use faiss_db
   db.query_analytics_by_faculty.find().pretty()
   db.query_analytics_by_faculty.countDocuments()
   ```
3. Trigger test queries trong Chat view
4. Check Spark logs: `docker logs spark-master`

### Embedding not working in React app

**Error**: iframes blocked hoặc không hiển thị

**Solutions**:

1. Verify **Public sharing enabled** for dashboard/questions
2. Check public URLs accessible trong browser
3. Verify CORS (Metabase allows embedding by default)
4. Check browser console for errors
5. Try disable browser extensions (ad blockers)

### Charts show old data

**Issue**: Data không update real-time

**Solutions**:

1. Reduce **Cache TTL** trong dashboard settings (set to 10-60 seconds)
2. Enable **Auto-refresh** trong dashboard
3. Manually click **Refresh** để force update
4. Clear Metabase cache: Settings → Admin → Troubleshooting → Clear cache

## 📚 Advanced Features

### Filters

Add filters to dashboard:

1. Edit dashboard
2. Click **Add filter**
3. Select filter type (Date, Faculty, etc.)
4. Connect filter to questions
5. Users can filter data interactively

### Alerts

Setup email alerts when metrics hit thresholds:

1. Open question
2. Click **Get alerts**
3. Configure conditions
4. Set email recipients

### SQL Queries

For advanced queries:

1. **New** → **SQL Query**
2. Write MongoDB aggregation pipeline in SQL format
3. Metabase translates to MongoDB queries

Example:

```sql
SELECT faculty, SUM(query_count) as total
FROM query_analytics_by_faculty
GROUP BY faculty
ORDER BY total DESC
LIMIT 10
```

## 🎯 Quick Reference

### Access Points

- **Metabase UI**: http://localhost:8090
- **API Health**: http://localhost:8000/analytics/health
- **Frontend**: http://localhost:5173 (vite dev server)

### Default Credentials

- **Email**: admin@example.com
- **Password**: (bạn đã set trong wizard)

### MongoDB Connection

- **Host**: host.docker.internal
- **Port**: 27017
- **Database**: faiss_db
- **Auth**: None

### Useful Commands

```bash
# Start Metabase
docker-compose -f docker-compose.charts.yml up -d

# Check logs
docker logs -f metabase-analytics

# Restart Metabase
docker-compose -f docker-compose.charts.yml restart metabase

# Stop Metabase
docker-compose -f docker-compose.charts.yml down

# Reset Metabase (delete data)
docker-compose -f docker-compose.charts.yml down -v
```

## 📖 Resources

- [Metabase Documentation](https://www.metabase.com/docs/latest/)
- [MongoDB Plugin Guide](https://www.metabase.com/data_sources/mongodb)
- [Embedding Guide](https://www.metabase.com/docs/latest/administration-guide/13-embedding.html)
- [Dashboard Best Practices](https://www.metabase.com/learn/dashboards/)

---

**💡 Tips**:

- Metabase tự động detect data types và suggest chart types
- Có thể export dashboards as PDF/PNG
- Support dark mode (Settings → Appearance)
- Có mobile responsive design
- Miễn phí hoàn toàn cho self-hosted (open-source)
