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
        self.mode = mode
        self.frame_skip = frame_skip
        self.enabled = enabled
    
    def is_batch_mode(self) -> bool:
        if self.mode == "batch":
            return True
        if self.mode == "realtime":
            return False
        if self.source.startswith(("rtsp://", "rtmp://", "http://")) or self.source.isdigit():
            return False
        return True


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
    heatmap_interval: int = 300
    detection_confidence: float = 0.5
    batch_frame_skip: int = 30
    
    # Feature Flags
    enable_gender_detection: bool = True
    enable_heatmap: bool = True
    enable_notifications: bool = False
    
    # Redis (for future async tasks)
    redis_url: str = "redis://localhost:6379/0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_cameras(self) -> List[CameraConfig]:
        try:
            cameras_data = json.loads(self.cameras_config)
            return [CameraConfig(**cam) for cam in cameras_data]
        except json.JSONDecodeError:
            return [CameraConfig(
                id="default",
                name="Main Camera", 
                zone="main",
                source=self.rtsp_url or "video.mp4"
            )]


settings = Settings()