import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Tuple, Optional, Dict, List
from datetime import datetime
import logging

from core import settings
from .gender_detection import HybridGenderDetector as GenderDetector

logger = logging.getLogger(__name__)


class FrameProcessor:
    """
    Processes individual frames for person detection and tracking
    """
    
    def __init__(self, model: YOLO, gender_detector: GenderDetector = None):
        self.model = model
        self.gender_detector = gender_detector
        
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            embedder_gpu=False
        )
        
        # Tracking state
        self.active_visitors: Dict[int, Dict] = {}
        self.completed_visitors: List[Dict] = []
        self.unique_visitor_count = 0
        self.gender_cache: Dict[int, Dict] = {}
        
        # Heatmap accumulator
        self.heatmap: Optional[np.ndarray] = None
    
    def process(self, frame: np.ndarray, video_time: float = 0) -> Tuple[int, Dict]:
        """
        Process a single frame
        Returns: (count, stats_dict)
        """
        stats = {
            "male": 0, "female": 0, "unknown": 0,
            "unique_visitors": self.unique_visitor_count,
            "active_visitors": len(self.active_visitors),
            "avg_dwell_time": 0
        }
        
        # Run YOLO detection
        results = self.model(frame, conf=settings.detection_confidence, classes=[0], verbose=False)
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(([x1, y1, x2, y2], conf, 'person'))
            
            # Update heatmap
            if settings.enable_heatmap:
                self._update_heatmap(frame, boxes)
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        active_track_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            active_track_ids.add(track_id)
            bbox = track.to_ltrb()
            
            # New visitor
            if track_id not in self.active_visitors:
                self.unique_visitor_count += 1
                self.active_visitors[track_id] = {
                    "first_seen_time": video_time,
                    "last_seen_time": video_time,
                    "first_seen": datetime.now(),
                    "last_seen": datetime.now(),
                    "detection_attempts": 0
                }
            else:
                self.active_visitors[track_id]["last_seen_time"] = video_time
                self.active_visitors[track_id]["last_seen"] = datetime.now()
            
            # Gender detection
            if settings.enable_gender_detection and self.gender_detector:
                self._detect_gender(frame, bbox, track_id)
        
        # Finalize visitors who left
        left_visitors = set(self.active_visitors.keys()) - active_track_ids
        for track_id in left_visitors:
            self._finalize_visitor(track_id)
        
        # Update stats
        stats["unique_visitors"] = self.unique_visitor_count
        stats["active_visitors"] = len(self.active_visitors)
        
        for track_id in active_track_ids:
            cached = self.gender_cache.get(track_id, {})
            gender = cached.get("gender", "unknown")
            if gender == "male":
                stats["male"] += 1
            elif gender == "female":
                stats["female"] += 1
            else:
                stats["unknown"] += 1
        
        if self.completed_visitors:
            total_dwell = sum(v["dwell_time"] for v in self.completed_visitors)
            stats["avg_dwell_time"] = round(total_dwell / len(self.completed_visitors), 1)
        
        return len(detections), stats
    
    def _detect_gender(self, frame: np.ndarray, bbox, track_id: int):
        """Detect gender for a tracked person"""
        cached = self.gender_cache.get(track_id)
        visitor = self.active_visitors.get(track_id, {})
        
        should_detect = (
            cached is None or 
            (cached.get("confidence", 0) < 0.6 and visitor.get("detection_attempts", 0) < 5)
        )
        
        if should_detect:
            self.active_visitors[track_id]["detection_attempts"] = visitor.get("detection_attempts", 0) + 1
            gender, method, confidence = self.gender_detector.detect(frame, bbox)
            
            if cached is None or confidence > cached.get("confidence", 0):
                self.gender_cache[track_id] = {
                    "gender": gender,
                    "method": method,
                    "confidence": confidence
                }
    
    def _finalize_visitor(self, track_id: int):
        """Mark visitor as completed and calculate dwell time"""
        if track_id not in self.active_visitors:
            return
        
        visitor = self.active_visitors.pop(track_id)
        dwell_time = visitor["last_seen_time"] - visitor["first_seen_time"]
        
        cached = self.gender_cache.get(track_id, {})
        
        self.completed_visitors.append({
            "track_id": track_id,
            "first_seen": visitor["first_seen"],
            "last_seen": visitor["last_seen"],
            "dwell_time": dwell_time,
            "gender": cached.get("gender", "unknown"),
            "gender_confidence": cached.get("confidence", 0)
        })
    
    def _update_heatmap(self, frame: np.ndarray, boxes) -> None:
        """Update heatmap accumulator"""
        if self.heatmap is None:
            self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        if self.heatmap.shape != frame.shape[:2]:
            self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(self.heatmap, (cx, cy), 50, 1, -1)
        
        self.heatmap = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
    
    def finalize_all(self):
        """Finalize all active visitors"""
        for track_id in list(self.active_visitors.keys()):
            self._finalize_visitor(track_id)
    
    def get_visitor_stats(self) -> Dict:
        """Get comprehensive visitor statistics"""
        all_visitors = self.completed_visitors.copy()
        
        # Include active visitors - use video time for dwell calculation
        for track_id, visitor in self.active_visitors.items():
            dwell = visitor.get("last_seen_time", 0) - visitor.get("first_seen_time", 0)
            cached = self.gender_cache.get(track_id, {})
            all_visitors.append({
                "track_id": track_id,
                "dwell_time": dwell,
                "gender": cached.get("gender", "unknown"),
                "is_active": True
            })
        
        if not all_visitors:
            return {
                "unique_visitors": 0,
                "active_visitors": 0,
                "avg_dwell_time": 0,
                "min_dwell_time": 0,
                "max_dwell_time": 0,
                "gender_breakdown": {"male": 0, "female": 0, "unknown": 0}
            }
        
        dwell_times = [v["dwell_time"] for v in all_visitors]
        genders = [v["gender"] for v in all_visitors]
        
        return {
            "unique_visitors": self.unique_visitor_count,
            "active_visitors": len(self.active_visitors),
            "completed_visitors": len(self.completed_visitors),
            "avg_dwell_time": round(sum(dwell_times) / len(dwell_times), 1),
            "min_dwell_time": round(min(dwell_times), 1),
            "max_dwell_time": round(max(dwell_times), 1),
            "gender_breakdown": {
                "male": genders.count("male"),
                "female": genders.count("female"),
                "unknown": genders.count("unknown") + genders.count(None)
            }
        }
    
    def reset(self):
        """Reset all tracking state"""
        self.tracker = DeepSort(
            max_age=30, n_init=3, max_cosine_distance=0.3,
            nn_budget=100, embedder="mobilenet", embedder_gpu=False
        )
        self.active_visitors = {}
        self.completed_visitors = []
        self.unique_visitor_count = 0
        self.gender_cache = {}
        self.heatmap = None
        
        if self.gender_detector:
            self.gender_detector.reset_stats()