#!/bin/bash

# EC2 Deployment Script for Retail Analytics Backend
set -e

echo "🚀 Deploying Retail Analytics Backend..."

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

# Create app directory
APP_DIR="/opt/retail-analytics"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR
cd $APP_DIR

# Copy application files (or git clone)
# Assuming files are already in current directory

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file (add your actual values)
cat > .env << EOF
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=retail-analytics-heatmaps
S3_REGION=us-east-1
RTSP_URL=rtsp://your-camera-url:554/stream
DATABASE_URL=sqlite+aiosqlite:///./analytics.db
CORS_ORIGINS=["*"]
LOG_LEVEL=INFO
EOF

echo "⚙️  Creating systemd service..."

# Create systemd service file
sudo tee /etc/systemd/system/retail-analytics.service > /dev/null << EOF
[Unit]
Description=Retail Video Analytics API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable retail-analytics
sudo systemctl start retail-analytics

echo "✅ Deployment complete!"
echo "📊 Check status: sudo systemctl status retail-analytics"
echo "📝 View logs: sudo journalctl -u retail-analytics -f"
echo "🌐 API running at: http://$(curl -s ifconfig.me):8000"
