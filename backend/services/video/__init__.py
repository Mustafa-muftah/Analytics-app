from .camera_manager import CameraManager, CameraInstance, camera_manager
from .processor import FrameProcessor
from .gender_detection import GenderDetector, PersonAttributesClassifier

__all__ = [
    'CameraManager', 'CameraInstance', 'camera_manager',
    'FrameProcessor', 'GenderDetector', 'PersonAttributesClassifier'
]