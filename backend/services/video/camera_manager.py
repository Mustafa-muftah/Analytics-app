import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import os
import logging

import boto3
from PIL import Image
from io import BytesIO

from core import settings, CameraConfig
from .processor import FrameProcessor
from .gender_detection import GenderDetector

logger = logging.getLogger(__name__)


class CameraInstance:
    """
    Manages a single camera/video source
    """
    
    def __init__(self, config: CameraConfig, processor: FrameProcessor, s3_client=None):
        self.config = config
        self.camera_id = config.id
        self.processor = processor
        self.s3_client = s3_client
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 30.0
        self.batch_completed = False
        
        logger.info(f"CameraInstance created: {self.camera_id}")
    
    def connect(self) -> bool:
        """Connect to video source"""
        try:
            if self.cap is not None:
                self.cap.release()
            
            source = self.config.source
            if source.isdigit():
                source = int(source)
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                logger.error(f"[{self.camera_id}] Failed to open: {source}")
                return False
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            if self.config.is_batch_mode():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"[{self.camera_id}] Batch: {self.total_frames} frames, {self.fps:.1f} FPS")
            
            return True
        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            return False
    
    def process_frame(self) -> Tuple[int, Optional[np.ndarray], Dict]:
        """Process next frame from video source"""
        stats = {"skipped": False}
        
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return 0, None, stats
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.config.is_batch_mode():
                self.processor.finalize_all()
                self.batch_completed = True
                logger.info(f"[{self.camera_id}] Batch completed")
                return -1, None, stats
            else:
                self.connect()
                return 0, None, stats
        
        self.frame_count += 1
        
        # Skip frames in batch mode
        if self.config.is_batch_mode():
            if self.frame_count % self.config.frame_skip != 0:
                return 0, None, {"skipped": True}
        
        video_time = self.frame_count / self.fps
        count, stats = self.processor.process(frame, video_time)
        
        return count, frame, stats
    
    def get_progress(self) -> Dict:
        """Get processing progress"""
        if self.total_frames == 0:
            return {
                "progress_percent": 0,
                "frame_count": self.frame_count,
                "total_frames": 0,
                "batch_completed": self.batch_completed
            }
        
        return {
            "progress_percent": min(100, int((self.frame_count / self.total_frames) * 100)),
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "unique_visitors": self.processor.unique_visitor_count,
            "batch_completed": self.batch_completed
        }
    
    def generate_heatmap(self) -> Tuple[Optional[str], float]:
        """Generate and save heatmap image"""
        start = time.time()
        
        if self.processor.heatmap is None:
            return None, 0
        
        try:
            heatmap_norm = cv2.normalize(self.processor.heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
            image = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processing_time = time.time() - start
            
            # Upload to S3 or save locally
            if self.s3_client and settings.s3_bucket:
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                buffer.seek(0)
                
                key = f"heatmaps/{self.camera_id}/heatmap_{timestamp}.png"
                self.s3_client.upload_fileobj(buffer, settings.s3_bucket, key)
                url = f"https://{settings.s3_bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"
                return url, processing_time
            else:
                local_dir = f"heatmaps/{self.camera_id}"
                os.makedirs(local_dir, exist_ok=True)
                path = f"{local_dir}/heatmap_{timestamp}.png"
                image.save(path)
                return path, processing_time
        except Exception as e:
            logger.error(f"[{self.camera_id}] Heatmap error: {e}")
            return None, 0
    
    def reset(self):
        """Reset camera state"""
        self.frame_count = 0
        self.batch_completed = False
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.processor.reset()
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        logger.info(f"[{self.camera_id}] Cleaned up")


class CameraManager:
    """
    Manages all cameras and shared resources
    """
    
    def __init__(self):
        self.cameras: Dict[str, CameraInstance] = {}
        self.model: Optional[YOLO] = None
        self.gender_detector: Optional[GenderDetector] = None
        self.s3_client = None
        self._initialized = False
    
    def initialize(self):
        """Initialize shared resources"""
        if self._initialized:
            return
        
        try:
            # Load YOLO model
            self.model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded")
            
            # Initialize gender detector
            if settings.enable_gender_detection:
                self.gender_detector = GenderDetector("models/person_attributes.xml")
            
            # Initialize S3
            if settings.aws_access_key_id and settings.s3_bucket:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.s3_region
                )
                logger.info("S3 client initialized")
            
            self._initialized = True
        except Exception as e:
            logger.error(f"CameraManager init failed: {e}")
            raise
    
    def add_camera(self, config: CameraConfig) -> CameraInstance:
        """Add a camera to management"""
        if not self._initialized:
            self.initialize()
        
        processor = FrameProcessor(self.model, self.gender_detector)
        camera = CameraInstance(config, processor, self.s3_client)
        self.cameras[config.id] = camera
        
        logger.info(f"Camera added: {config.id}")
        return camera
    
    def get_camera(self, camera_id: str) -> Optional[CameraInstance]:
        return self.cameras.get(camera_id)
    
    def get_all_cameras(self) -> List[CameraInstance]:
        return list(self.cameras.values())
    
    def get_camera_configs(self) -> List[Dict]:
        return [
            {
                "id": cam.camera_id,
                "name": cam.config.name,
                "zone": cam.config.zone,
                "mode": cam.config.mode,
                "source": cam.config.source
            }
            for cam in self.cameras.values()
        ]
    
    def load_from_settings(self):
        """Load cameras from settings"""
        for config in settings.get_cameras():
            self.add_camera(config)
    
    def cleanup(self):
        """Cleanup all cameras"""
        for camera in self.cameras.values():
            camera.cleanup()
        logger.info("CameraManager cleaned up")


# Singleton instance
camera_manager = CameraManager()