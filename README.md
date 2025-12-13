# Retail Video Analytics System v2.0

AI-powered retail analytics platform using computer vision for visitor tracking, gender detection, dwell time analysis, and traffic heatmaps.

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **Unique Visitor Counting** | DeepSORT tracking - no double counting |
| **Gender Detection** | InsightFace AI - male/female classification |
| **Dwell Time Analysis** | Track how long visitors stay |
| **Traffic Heatmap** | Visual heat patterns of movement |
| **Peak Hours Analysis** | Identify busiest times |
| **Multi-Camera Support** | Scale from 1 to many cameras |
| **Batch Processing** | Fast analysis of recorded video |
| **PDF Reports** | Export comprehensive analytics |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- 4GB+ RAM recommended

### Installation

```bash
# Clone/Extract the project
cd retail-analytics-mvp

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

---

## 📋 Demo Mode (Video File Analysis)

**Use this for customer demos with recorded video.**

### Step 1: Configure Demo

```bash
cd backend
cp .env.example .env
```

Edit `.env`:
```env
# Demo configuration - single camera, batch mode
CAMERAS_CONFIG='[{"id": "main", "name": "Store Camera", "zone": "main", "source": "customer_video.mp4", "mode": "batch", "frame_skip": 30}]'

# Leave AWS empty for local heatmap storage
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET=
```

### Step 2: Add Customer Video

```bash
# Place the customer's video in backend folder
cp /path/to/customer_video.mp4 backend/video.mp4
```

**Supported formats:** MP4, AVI, MOV, MKV

### Step 3: Start Services

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Step 4: Process Video

1. Open http://localhost:5173
2. Go to **Reports** tab
3. Click **"Start Batch Processing"**
4. Wait for progress bar to complete (~2-5 min for 20min video)
5. View analytics on **Dashboard**
6. Click **"Generate PDF Report"** to export

### Demo Timeline

| Video Length | Processing Time | Output |
|--------------|-----------------|--------|
| 5 min | ~30 sec | Quick demo |
| 20 min | ~2-3 min | Full analytics |
| 1 hour | ~8-10 min | Comprehensive data |

---

## 🏭 Production Mode (Live Cameras)

**Use this for actual retail deployment.**

### Step 1: Configure Cameras

Edit `backend/.env`:

```env
# Single camera production
CAMERAS_CONFIG='[{
  "id": "entrance",
  "name": "Main Entrance",
  "zone": "entrance",
  "source": "rtsp://admin:password@192.168.1.100:554/stream",
  "mode": "realtime"
}]'

# Multi-camera production
CAMERAS_CONFIG='[
  {"id": "cam_01", "name": "Entrance", "zone": "entrance", "source": "rtsp://admin:pass@192.168.1.100:554/stream", "mode": "realtime"},
  {"id": "cam_02", "name": "Aisle A", "zone": "aisle_a", "source": "rtsp://admin:pass@192.168.1.101:554/stream", "mode": "realtime"},
  {"id": "cam_03", "name": "Aisle B", "zone": "aisle_b", "source": "rtsp://admin:pass@192.168.1.102:554/stream", "mode": "realtime"},
  {"id": "cam_04", "name": "Cashier", "zone": "cashier", "source": "rtsp://admin:pass@192.168.1.103:554/stream", "mode": "realtime"}
]'

# AWS for heatmap storage (optional but recommended)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your-bucket-name
S3_REGION=us-east-1
```

### Step 2: Camera Source Options

| Type | Example | Use Case |
|------|---------|----------|
| RTSP | `rtsp://admin:pass@ip:554/stream` | IP cameras |
| HTTP | `http://camera/live.m3u8` | Web streams |
| Cloud | `https://s3.amazonaws.com/bucket/video.mp4` | Cloud recordings |
| Webcam | `0` or `1` | USB cameras |
| File | `recording.mp4` | Batch analysis |

### Step 3: Deploy Backend

**Option A: Direct (Development)**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Option B: Production (Gunicorn)**
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

**Option C: Systemd Service**
```bash
# /etc/systemd/system/retail-analytics.service
[Unit]
Description=Retail Analytics API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/retail-analytics/backend
Environment="PATH=/opt/retail-analytics/backend/venv/bin"
ExecStart=/opt/retail-analytics/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Step 4: Deploy Frontend

**Option A: Vercel (Recommended)**
```bash
cd frontend
npm install -g vercel
vercel --prod
```

**Option B: Nginx Static**
```bash
cd frontend
npm run build
# Copy dist/ to nginx html folder
```

---

## 📊 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/cameras` | GET | List all cameras |
| `/api/count` | GET | Current people count |
| `/api/peak-hours` | GET | Hourly analytics |
| `/api/gender-stats` | GET | Gender distribution |
| `/api/visitor-stats` | GET | Unique visitors & dwell time |
| `/api/dwell-time` | GET | Dwell time statistics |
| `/api/zone-stats` | GET | Zone comparison |
| `/api/heatmap` | GET | Latest heatmap URL |
| `/api/report/summary` | GET | Full report data |

### Batch Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/batch/start/{camera_id}` | POST | Start batch processing |
| `/api/batch/progress/{camera_id}` | GET | Get progress % |
| `/api/batch/jobs` | GET | List processing jobs |

### Query Parameters

All endpoints support `?camera_id=X` to filter by camera.

---

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERAS_CONFIG` | See example | JSON array of cameras |
| `DATABASE_URL` | `sqlite+aiosqlite:///./analytics.db` | Database connection |
| `DETECTION_CONFIDENCE` | `0.5` | YOLO confidence threshold |
| `HEATMAP_INTERVAL` | `300` | Heatmap generation interval (sec) |
| `BATCH_FRAME_SKIP` | `30` | Process every Nth frame in batch |
| `ENABLE_GENDER_DETECTION` | `true` | Enable/disable gender detection |
| `ENABLE_HEATMAP` | `true` | Enable/disable heatmap |

### Camera Config Options

```json
{
  "id": "unique_id",
  "name": "Display Name",
  "zone": "zone_name",
  "source": "video.mp4",
  "mode": "auto",
  "frame_skip": 30,
  "enabled": true
}
```

---

## 🔧 Troubleshooting

### Common Issues

**"Camera not connected"**
- Check RTSP URL is correct
- Verify network connectivity
- Try opening stream in VLC first

**"No faces detected"**
- Ensure camera captures faces (not backs)
- Adjust `DETECTION_CONFIDENCE` lower
- Check lighting conditions

**"Batch processing slow"**
- Increase `BATCH_FRAME_SKIP` (e.g., 60)
- Use smaller resolution video
- Check CPU usage

**"Heatmap not generating"**
- Wait for heatmap interval (5 min default)
- Check S3 credentials or use local storage
- Verify write permissions

---

## 📈 Scaling Guide

### Small Retail (1-2 cameras)
- Single server (t3.medium)
- SQLite database
- Local heatmap storage

### Medium Retail (3-5 cameras)
- Larger server (t3.large)
- PostgreSQL database
- S3 for heatmaps

### Large Retail (5+ cameras)
- Multiple processing servers
- Load balancer
- PostgreSQL + Redis
- S3 + CloudFront

---

## 📄 License

MIT License - Free for commercial use.
