from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List
import json


class CameraConfig:
    """Configuration for a single camera"""
    def __init__(self, id: str, name: str, zone: str, source: str, 
                 mode: str = "auto", frame_skip: int = 30, enabled: bool = True):
        self.id = id
        self.name = name
        self.zone = zone
        self.source = source
        self.mode = mode  # "realtime", "batch", "auto"
        self.frame_skip = frame_skip  # For batch mode: process every Nth frame
        self.enabled = enabled
    
    def is_batch_mode(self) -> bool:
        """Determine if batch mode should be used"""
        if self.mode == "batch":
            return True
        if self.mode == "realtime":
            return False
        # Auto-detect: file = batch, stream = realtime
        if self.source.startswith(("rtsp://", "rtmp://", "http://")) or self.source.isdigit():
            return False
        return True  # Local file = batch mode


class Settings(BaseSettings):
    # AWS Configuration
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    
    # Camera Configuration (JSON string)
    cameras_config: str = '[{"id": "default", "name": "Main Camera", "zone": "main", "source": "video.mp4", "mode": "auto"}]'
    
    # Legacy single camera support
    rtsp_url: str = ""
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./analytics.db"
    
    # API Configuration
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    
    # Analytics Settings
    heatmap_interval: int = 300  # 5 minutes in seconds
    detection_confidence: float = 0.5
    batch_frame_skip: int = 30  # Process every Nth frame in batch mode
    
    # Processing
    enable_gender_detection: bool = True
    enable_heatmap: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_cameras(self) -> List[CameraConfig]:
        """Parse camera configuration from JSON string"""
        try:
            cameras_data = json.loads(self.cameras_config)
            return [CameraConfig(**cam) for cam in cameras_data]
        except json.JSONDecodeError:
            # Fallback to legacy single camera
            return [CameraConfig(
                id="default",
                name="Main Camera", 
                zone="main",
                source=self.rtsp_url or "video.mp4"
            )]


settings = Settings()
