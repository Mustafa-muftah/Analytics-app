import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional, Dict, List
import logging
from datetime import datetime, timedelta
import boto3
from io import BytesIO
from PIL import Image
from config import settings, CameraConfig
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os

# Try to import InsightFace, but it's optional now
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    FaceAnalysis = None

# Import person attributes classifier (ChatGPT solution)
from person_attributes_classifier import (
    PersonAttributesClassifier, 
    HybridGenderClassifier as PAHybridClassifier,
    ONNX_AVAILABLE,
    OPENVINO_AVAILABLE
)

logger = logging.getLogger(__name__)


class BodyGenderClassifier:
    def __init__(self):
        self.initialized = True
        logger.info("Body gender classifier initialized")
    
    def predict(self, person_crop: np.ndarray) -> Tuple[str, float]:
        try:
            if person_crop is None or person_crop.size == 0:
                return "unknown", 0.0
            
            h, w = person_crop.shape[:2]
            if h < 30 or w < 15:
                return "unknown", 0.0
            
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            
            male_score = 0.0
            female_score = 0.0
            
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
            v_hist = v_hist.flatten() / (v_hist.sum() + 1e-6)
            
            pink_ratio = h_hist[140:170].sum()
            if pink_ratio > 0.1:
                female_score += 0.3
            
            red_ratio = h_hist[0:10].sum() + h_hist[170:180].sum()
            if red_ratio > 0.15:
                female_score += 0.2
            
            blue_ratio = h_hist[100:130].sum()
            dark_value = v_hist[0:100].sum()
            if blue_ratio > 0.2 and dark_value > 0.4:
                male_score += 0.25
            
            high_sat = s_hist[150:256].sum()
            high_val = v_hist[180:256].sum()
            if high_sat > 0.3 and high_val > 0.3:
                female_score += 0.2
            
            low_sat = s_hist[0:50].sum()
            if low_sat > 0.5:
                male_score += 0.2
            
            aspect_ratio = h / w
            if aspect_ratio < 2.0:
                male_score += 0.15
            elif aspect_ratio > 2.5:
                female_score += 0.15
            
            upper_region = person_crop[0:h//3, :]
            if upper_region.size > 0:
                upper_hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
                upper_h = cv2.calcHist([upper_hsv], [0], None, [180], [0, 180])
                upper_h = upper_h.flatten() / (upper_h.sum() + 1e-6)
                
                warm_colors = upper_h[0:30].sum() + upper_h[150:180].sum()
                if warm_colors > 0.2:
                    female_score += 0.15
            
            total_score = male_score + female_score
            if total_score < 0.2:
                return "unknown", 0.3
            
            if male_score > female_score:
                confidence = min(0.8, 0.5 + (male_score - female_score))
                return "male", confidence
            elif female_score > male_score:
                confidence = min(0.8, 0.5 + (female_score - male_score))
                return "female", confidence
            else:
                return "unknown", 0.4
                
        except Exception as e:
            logger.debug(f"Body gender classification error: {e}")
            return "unknown", 0.0


class HybridGenderDetector:
    """
    Gender detector that uses:
    1. InsightFace (if available and face visible) - highest accuracy for faces
    2. PersonAttributesClassifier from OpenVINO (body-based, ~92% accuracy)
    3. Basic heuristics fallback
    """
    
    def __init__(self, face_analyzer=None, model_path: str = None):
        self.face_analyzer = face_analyzer
        
        # Use person attributes classifier (ChatGPT solution)
        self.pa_classifier = PAHybridClassifier(model_path)
        
        self.detection_stats = {"face": 0, "model": 0, "fallback": 0, "unknown": 0}
        
        model_status = "loaded" if self.pa_classifier.pa_classifier.initialized else "not found"
        logger.info(f"Hybrid gender detector initialized (InsightFace: {face_analyzer is not None}, PA Model: {model_status})")
    
    def detect(self, frame: np.ndarray, bbox, retry_count: int = 0) -> Tuple[str, str, float]:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return "unknown", "none", 0.0
        
        # Strategy 1: InsightFace on head region (highest accuracy for visible faces)
        if self.face_analyzer is not None:
            try:
                crop_h = person_crop.shape[0]
                head_region = person_crop[0:int(crop_h * 0.4), :]
                
                if head_region.size > 0:
                    faces = self.face_analyzer.get(head_region)
                    if faces and len(faces) > 0:
                        face = faces[0]
                        gender_val = face.get('gender', None)
                        if gender_val is not None:
                            gender = "male" if gender_val == 1 else "female"
                            confidence = 0.95
                            self.detection_stats["face"] += 1
                            logger.debug(f"Gender detected via InsightFace: {gender}")
                            return gender, "face", confidence
            except Exception as e:
                logger.debug(f"InsightFace detection failed: {e}")
        
        # Strategy 2: PersonAttributesClassifier (body-based, ~92% accuracy)
        gender, method, confidence = self.pa_classifier.predict(frame, bbox)
        
        if gender != "unknown":
            if method == "model":
                self.detection_stats["model"] += 1
            else:
                self.detection_stats["fallback"] += 1
            logger.debug(f"Gender detected via {method}: {gender} (conf: {confidence:.2f})")
            return gender, method, confidence
        
        self.detection_stats["unknown"] += 1
        return "unknown", "none", 0.0
    
    def get_stats(self) -> Dict:
        total = sum(self.detection_stats.values())
        if total == 0:
            return {"face_pct": 0, "model_pct": 0, "fallback_pct": 0, "unknown_pct": 0, "total": 0}
        
        return {
            "face_pct": round(self.detection_stats["face"] / total * 100, 1),
            "model_pct": round(self.detection_stats["model"] / total * 100, 1),
            "fallback_pct": round(self.detection_stats["fallback"] / total * 100, 1),
            "unknown_pct": round(self.detection_stats["unknown"] / total * 100, 1),
            "total": total
        }
    
    def reset_stats(self):
        self.detection_stats = {"face": 0, "model": 0, "fallback": 0, "unknown": 0}
        self.pa_classifier.reset_stats()


class CameraAnalytics:
    def __init__(self, camera_config: CameraConfig, model: YOLO, gender_detector: HybridGenderDetector, s3_client):
        self.config = camera_config
        self.camera_id = camera_config.id
        self.model = model
        self.gender_detector = gender_detector
        self.s3_client = s3_client
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.heatmap: Optional[np.ndarray] = None
        self.is_running = False
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 30.0  # Default FPS, will be updated from video
        
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            embedder_gpu=False
        )
        
        self.active_visitors: Dict[int, Dict] = {}
        self.completed_visitors: List[Dict] = []
        self.unique_visitor_count = 0
        self.gender_cache: Dict[int, Dict] = {}
        
        self.batch_stats = {
            "people_counts": [],
            "gender_stats": [],
            "processed_frames": 0,
            "unique_visitors": 0,
            "total_dwell_time": 0
        }
        
        logger.info(f"CameraAnalytics initialized for {self.camera_id} ({camera_config.name}) with hybrid gender detection")
    
    def connect(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
            
            source = self.config.source
            if source.isdigit():
                source = int(source)
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                logger.error(f"[{self.camera_id}] Failed to open source: {source}")
                return False
            
            # Get video FPS for accurate dwell time calculation
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0  # Default fallback
            
            if self.config.is_batch_mode():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = self.total_frames / self.fps
                logger.info(f"[{self.camera_id}] Batch mode: {self.total_frames} frames, {self.fps:.1f} FPS, {video_duration:.1f}s duration")
            
            logger.info(f"[{self.camera_id}] Connected to source: {source}")
            return True
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            return False
    
    def _get_video_timestamp(self) -> float:
        """Get current video timestamp in seconds based on frame count"""
        if self.config.is_batch_mode():
            return self.frame_count / self.fps
        else:
            # Real-time mode uses wall clock
            return time.time()
    
    def process_frame(self) -> Tuple[int, Optional[np.ndarray], Dict]:
        stats = {
            "male": 0, 
            "female": 0, 
            "unknown": 0,
            "unique_visitors": self.unique_visitor_count,
            "active_visitors": len(self.active_visitors),
            "avg_dwell_time": 0,
            "gender_methods": {"face": 0, "body": 0}
        }
        
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return 0, None, stats
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.config.is_batch_mode():
                    self._finalize_all_visitors()
                    logger.info(f"[{self.camera_id}] Batch completed. Unique visitors: {self.unique_visitor_count}")
                    
                    if self.gender_detector:
                        gd_stats = self.gender_detector.get_stats()
                        logger.info(f"[{self.camera_id}] Gender detection: {gd_stats['face_pct']}% face, {gd_stats['body_pct']}% body, {gd_stats['unknown_pct']}% unknown")
                    
                    return -1, None, stats
                else:
                    logger.warning(f"[{self.camera_id}] Frame read failed, reconnecting...")
                    self.connect()
                    return 0, None, stats
            
            self.frame_count += 1
            
            # Use video timestamp for batch mode, wall clock for real-time
            current_video_time = self._get_video_timestamp()
            
            if self.config.is_batch_mode():
                if self.frame_count % self.config.frame_skip != 0:
                    return 0, None, {"skipped": True}
            
            results = self.model(frame, conf=settings.detection_confidence, classes=[0], verbose=False)
            
            count = 0
            detections = []
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                count = len(boxes)
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append(([x1, y1, x2, y2], conf, 'person'))
                
                if settings.enable_heatmap:
                    self._update_heatmap(frame, boxes)
            
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            active_track_ids = set()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                active_track_ids.add(track_id)
                bbox = track.to_ltrb()
                
                if track_id not in self.active_visitors:
                    self.unique_visitor_count += 1
                    self.active_visitors[track_id] = {
                        "first_seen_video_time": current_video_time,  # Video timestamp in seconds
                        "last_seen_video_time": current_video_time,
                        "first_seen": datetime.now(),  # Wall clock for display
                        "last_seen": datetime.now(),
                        "gender": None,
                        "gender_method": None,
                        "gender_confidence": 0,
                        "positions": [(int(bbox[0] + bbox[2])//2, int(bbox[1] + bbox[3])//2)],
                        "frame_count": 1,
                        "detection_attempts": 0
                    }
                    logger.debug(f"[{self.camera_id}] New visitor: Track {track_id} at video time {current_video_time:.1f}s")
                else:
                    self.active_visitors[track_id]["last_seen_video_time"] = current_video_time
                    self.active_visitors[track_id]["last_seen"] = datetime.now()
                    self.active_visitors[track_id]["positions"].append(
                        (int(bbox[0] + bbox[2])//2, int(bbox[1] + bbox[3])//2)
                    )
                    self.active_visitors[track_id]["frame_count"] += 1
                
                if settings.enable_gender_detection and self.gender_detector:
                    cached = self.gender_cache.get(track_id)
                    
                    should_detect = (
                        cached is None or 
                        (cached["confidence"] < 0.6 and self.active_visitors[track_id]["detection_attempts"] < 5)
                    )
                    
                    if should_detect:
                        self.active_visitors[track_id]["detection_attempts"] += 1
                        retry = self.active_visitors[track_id]["detection_attempts"] - 1
                        
                        gender, method, confidence = self.gender_detector.detect(frame, bbox, retry_count=retry)
                        
                        if cached is None or confidence > cached["confidence"]:
                            self.gender_cache[track_id] = {
                                "gender": gender,
                                "method": method,
                                "confidence": confidence
                            }
                            self.active_visitors[track_id]["gender"] = gender
                            self.active_visitors[track_id]["gender_method"] = method
                            self.active_visitors[track_id]["gender_confidence"] = confidence
            
            left_visitors = set(self.active_visitors.keys()) - active_track_ids
            for track_id in left_visitors:
                self._finalize_visitor(track_id)
            
            stats["unique_visitors"] = self.unique_visitor_count
            stats["active_visitors"] = len(self.active_visitors)
            
            for track_id in active_track_ids:
                cached = self.gender_cache.get(track_id)
                if cached:
                    gender = cached["gender"]
                    method = cached["method"]
                    if gender == "male":
                        stats["male"] += 1
                    elif gender == "female":
                        stats["female"] += 1
                    else:
                        stats["unknown"] += 1
                    
                    if method == "face":
                        stats["gender_methods"]["face"] += 1
                    elif method == "body":
                        stats["gender_methods"]["body"] += 1
                else:
                    stats["unknown"] += 1
            
            if self.completed_visitors:
                total_dwell = sum(v["dwell_time"] for v in self.completed_visitors)
                stats["avg_dwell_time"] = round(total_dwell / len(self.completed_visitors), 1)
            
            return count, frame, stats
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] Frame processing error: {e}")
            return 0, None, stats
    
    def _finalize_visitor(self, track_id: int):
        if track_id not in self.active_visitors:
            return
        
        visitor = self.active_visitors.pop(track_id)
        
        # Calculate dwell time from VIDEO timestamps (not wall clock)
        if self.config.is_batch_mode():
            dwell_time = visitor["last_seen_video_time"] - visitor["first_seen_video_time"]
        else:
            # Real-time mode uses wall clock
            dwell_time = (visitor["last_seen"] - visitor["first_seen"]).total_seconds()
        
        cached = self.gender_cache.get(track_id, {})
        
        self.completed_visitors.append({
            "track_id": track_id,
            "first_seen": visitor["first_seen"],
            "last_seen": visitor["last_seen"],
            "dwell_time": dwell_time,
            "gender": cached.get("gender", "unknown"),
            "gender_method": cached.get("method", "none"),
            "gender_confidence": cached.get("confidence", 0),
            "path_length": len(visitor["positions"])
        })
        
        logger.debug(f"[{self.camera_id}] Visitor {track_id} left. Dwell: {dwell_time:.1f}s, Gender: {cached.get('gender', 'unknown')}")
    
    def _finalize_all_visitors(self):
        for track_id in list(self.active_visitors.keys()):
            self._finalize_visitor(track_id)
    
    def _update_heatmap(self, frame: np.ndarray, boxes) -> None:
        if self.heatmap is None:
            self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        if self.heatmap.shape != frame.shape[:2]:
            self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(self.heatmap, (center_x, center_y), 50, 1, -1)
        
        self.heatmap = cv2.GaussianBlur(self.heatmap, (51, 51), 0)
    
    def generate_heatmap_image(self) -> Tuple[Optional[str], float]:
        start_time = time.time()
        
        if self.heatmap is None:
            return None, 0
        
        try:
            heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            image = Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
            
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.s3_client and settings.s3_bucket:
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                buffer.seek(0)
                
                key = f"heatmaps/{self.camera_id}/heatmap_{timestamp}.png"
                self.s3_client.upload_fileobj(buffer, settings.s3_bucket, key)
                
                url = f"https://{settings.s3_bucket}.s3.{settings.s3_region}.amazonaws.com/{key}"
                logger.info(f"[{self.camera_id}] Heatmap uploaded to S3: {key}")
                return url, processing_time
            else:
                local_dir = f"heatmaps/{self.camera_id}"
                os.makedirs(local_dir, exist_ok=True)
                local_path = f"{local_dir}/heatmap_{timestamp}.png"
                image.save(local_path)
                logger.info(f"[{self.camera_id}] Heatmap saved locally: {local_path}")
                return local_path, processing_time
                
        except Exception as e:
            logger.error(f"[{self.camera_id}] Heatmap generation error: {e}")
            return None, 0
    
    def get_progress(self) -> Dict:
        if self.total_frames == 0:
            return {"progress_percent": 0, "frame_count": self.frame_count, "total_frames": 0}
        
        progress = min(100, int((self.frame_count / self.total_frames) * 100))
        return {
            "progress_percent": progress,
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "unique_visitors": self.unique_visitor_count
        }
    
    def reset(self):
        self.frame_count = 0
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.heatmap = None
        self.active_visitors = {}
        self.completed_visitors = []
        self.unique_visitor_count = 0
        self.gender_cache = {}
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            embedder_gpu=False
        )
        self.batch_stats = {
            "people_counts": [],
            "gender_stats": [],
            "processed_frames": 0,
            "unique_visitors": 0,
            "total_dwell_time": 0
        }
        # Reset gender detector stats
        if self.gender_detector:
            self.gender_detector.reset_stats()
    
    def get_visitor_stats(self) -> Dict:
        all_visitors = self.completed_visitors.copy()
        
        current_video_time = self._get_video_timestamp()
        for track_id, visitor in self.active_visitors.items():
            if self.config.is_batch_mode():
                dwell = current_video_time - visitor["first_seen_video_time"]
            else:
                dwell = (datetime.now() - visitor["first_seen"]).total_seconds()
            
            cached = self.gender_cache.get(track_id, {})
            all_visitors.append({
                "track_id": track_id,
                "dwell_time": dwell,
                "gender": cached.get("gender", "unknown"),
                "gender_method": cached.get("method", "none"),
                "is_active": True
            })
        
        if not all_visitors:
            return {
                "unique_visitors": 0,
                "active_visitors": 0,
                "avg_dwell_time": 0,
                "min_dwell_time": 0,
                "max_dwell_time": 0,
                "gender_breakdown": {"male": 0, "female": 0, "unknown": 0},
                "detection_methods": {"face": 0, "body": 0}
            }
        
        dwell_times = [v["dwell_time"] for v in all_visitors]
        genders = [v["gender"] for v in all_visitors]
        methods = [v.get("gender_method", "none") for v in all_visitors]
        
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
            },
            "detection_methods": {
                "face": methods.count("face"),
                "body": methods.count("body")
            }
        }
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            logger.info(f"[{self.camera_id}] Resources released")


class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, CameraAnalytics] = {}
        self.model: Optional[YOLO] = None
        self.gender_detector: Optional[HybridGenderDetector] = None
        self.s3_client = None
        self._initialized = False
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            self.model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded")
            
            if settings.enable_gender_detection:
                face_analyzer = None
                
                # Try to load InsightFace if available
                if INSIGHTFACE_AVAILABLE:
                    try:
                        face_analyzer = None  # Disabled - InsightFace not compatible
                        # face_analyzer = FaceAnalysis(allowed_modules=['detection', 'genderage'])
                        face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
                        logger.info("InsightFace initialized for face detection")
                    except Exception as e:
                        logger.warning(f"InsightFace init failed: {e}")
                        face_analyzer = None
                else:
                    logger.info("InsightFace not available, using body-based detection")
                
                # Initialize hybrid detector with person attributes model
                self.gender_detector = HybridGenderDetector(
                    face_analyzer=face_analyzer,
                    model_path="models/person_attributes.xml"
                )
                logger.info(f"Gender detection ready (ONNX: {ONNX_AVAILABLE}, OpenVINO: {OPENVINO_AVAILABLE})")
            
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
            logger.error(f"CameraManager initialization failed: {e}")
            raise
    
    def add_camera(self, config: CameraConfig) -> CameraAnalytics:
        if not self._initialized:
            self.initialize()
        
        camera = CameraAnalytics(
            camera_config=config,
            model=self.model,
            gender_detector=self.gender_detector,
            s3_client=self.s3_client
        )
        
        self.cameras[config.id] = camera
        logger.info(f"Camera added: {config.id} ({config.name})")
        return camera
    
    def get_camera(self, camera_id: str) -> Optional[CameraAnalytics]:
        return self.cameras.get(camera_id)
    
    def get_all_cameras(self) -> List[CameraAnalytics]:
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
    
    def load_cameras_from_config(self):
        for cam_config in settings.get_cameras():
            self.add_camera(cam_config)
    
    def process_frame(self, camera_id: str) -> Tuple[int, Optional[np.ndarray], Dict]:
        camera = self.get_camera(camera_id)
        if camera:
            return camera.process_frame()
        return 0, None, {}
    
    def get_progress(self, camera_id: str) -> Dict:
        camera = self.get_camera(camera_id)
        if camera:
            return camera.get_progress()
        return {"progress_percent": 0}
    
    def cleanup(self):
        for camera in self.cameras.values():
            camera.cleanup()
        logger.info("CameraManager cleanup completed")


camera_manager = CameraManager()


class VideoAnalyticsCompat:
    def __init__(self):
        self.manager = camera_manager
    
    def initialize(self):
        self.manager.initialize()
        self.manager.load_cameras_from_config()
    
    def count_people(self) -> Tuple[int, Dict]:
        cameras = self.manager.get_all_cameras()
        if cameras:
            count, frame, stats = cameras[0].process_frame()
            return count, stats
        return 0, {}


video_analytics = VideoAnalyticsCompat()