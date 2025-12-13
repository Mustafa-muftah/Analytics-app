"""
Person Attributes Recognition using OpenVINO Model
Model: person-attributes-recognition-crossroad-0234
Source: OpenVINO Model Zoo

Attributes detected:
- is_male (gender)
- has_bag
- has_hat  
- has_longsleeves
- has_longpants
- has_longhair
- has_coat_jacket

Input: BGR image, resized to 80x160 (W x H)
Output: 7 probabilities
"""

import cv2
import numpy as np
import os
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# Try ONNX Runtime first
ONNX_AVAILABLE = False
OPENVINO_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("ONNX Runtime available")
except ImportError:
    pass

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
    logger.info("OpenVINO Runtime available")
except ImportError:
    pass


class PersonAttributesClassifier:
    """
    Person attributes classifier using OpenVINO's person-attributes-recognition-crossroad-0234 model.
    
    Supports both ONNX Runtime and OpenVINO IR formats.
    """
    
    # Model input size (width, height)
    INPUT_SIZE = (80, 160)
    
    # Attribute names in order
    ATTRIBUTES = [
        "is_male",
        "has_bag", 
        "has_hat",
        "has_longsleeves",
        "has_longpants",
        "has_longhair",
        "has_coat_jacket"
    ]
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to model file (.onnx or .xml)
                       If None, will search in default locations.
        """
        self.session = None
        self.compiled_model = None
        self.input_name = None
        self.output_name = None
        self.backend = None
        self.initialized = False
        
        # Default model paths to search
        self.default_paths = [
            "models/person_attributes.xml",
            "models/person_attributes.onnx",
            "models/person-attributes-recognition-crossroad-0234.xml",
        ]
        
        if model_path:
            self.default_paths.insert(0, model_path)
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self) -> bool:
        """Try to load the model from available paths."""
        
        for path in self.default_paths:
            if os.path.exists(path):
                if self._load_model(path):
                    return True
        
        logger.warning("Person attributes model not found. Gender detection will use fallback.")
        logger.warning(f"Searched paths: {self.default_paths}")
        return False
    
    def _load_model(self, model_path: str) -> bool:
        """Load model from file."""
        
        try:
            if model_path.endswith('.onnx') and ONNX_AVAILABLE:
                return self._load_onnx(model_path)
            elif model_path.endswith('.xml') and OPENVINO_AVAILABLE:
                return self._load_openvino(model_path)
            else:
                logger.warning(f"Unsupported model format or missing runtime: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return False
    
    def _load_onnx(self, model_path: str) -> bool:
        """Load ONNX model."""
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.backend = "onnx"
            self.initialized = True
            logger.info(f"Loaded ONNX model: {model_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX load error: {e}")
            return False
    
    def _load_openvino(self, model_path: str) -> bool:
        """Load OpenVINO IR model."""
        try:
            core = Core()
            model = core.read_model(model_path)
            self.compiled_model = core.compile_model(model, "CPU")
            self.input_name = list(self.compiled_model.inputs)[0]
            self.output_name = list(self.compiled_model.outputs)[0]
            self.backend = "openvino"
            self.initialized = True
            logger.info(f"Loaded OpenVINO model: {model_path}")
            return True
        except Exception as e:
            logger.error(f"OpenVINO load error: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: BGR image (any size)
            
        Returns:
            Preprocessed tensor of shape (1, 3, 160, 80)
        """
        # Resize to model input size (80 width, 160 height)
        resized = cv2.resize(image, self.INPUT_SIZE)
        
        # Convert to float32
        img = resized.astype(np.float32)
        
        # Transpose from HWC to CHW format
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_raw(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get raw attribute predictions.
        
        Args:
            image: BGR image (person crop)
            
        Returns:
            Array of 7 probabilities or None if failed
        """
        if not self.initialized:
            return None
        
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Run inference
            if self.backend == "onnx":
                outputs = self.session.run(None, {self.input_name: input_tensor})
                return outputs[0][0]
            elif self.backend == "openvino":
                result = self.compiled_model([input_tensor])
                return result[self.output_name][0]
            
        except Exception as e:
            logger.debug(f"Inference error: {e}")
            return None
    
    def predict_gender(self, image: np.ndarray, threshold: float = 0.5, 
                       unknown_threshold: float = 0.6) -> Tuple[str, float]:
        """
        Predict gender from person crop.
        
        Args:
            image: BGR image (full body person crop from YOLO)
            threshold: Decision threshold for male/female (default 0.5)
            unknown_threshold: Confidence below this returns "unknown" (default 0.6)
            
        Returns:
            (gender, confidence) where gender is "male", "female", or "unknown"
        """
        attrs = self.predict_raw(image)
        
        if attrs is None:
            return "unknown", 0.0
        
        is_male_score = float(attrs[0])
        
        # Determine gender
        if is_male_score > threshold:
            gender = "male"
            confidence = is_male_score
        else:
            gender = "female"
            confidence = 1.0 - is_male_score
        
        # Check confidence threshold
        if confidence < unknown_threshold:
            return "unknown", confidence
        
        return gender, confidence
    
    def predict_all_attributes(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Predict all 7 attributes.
        
        Args:
            image: BGR image (full body person crop)
            
        Returns:
            Dictionary of attribute names to probabilities
        """
        attrs = self.predict_raw(image)
        
        if attrs is None:
            return None
        
        return {name: float(prob) for name, prob in zip(self.ATTRIBUTES, attrs)}


class BodyHeuristicsClassifier:
    """
    Simple body heuristics based on color and shape.
    Used in combination with PA Model for voting.
    """
    
    def predict(self, person_crop: np.ndarray) -> Tuple[str, float]:
        """Predict gender using color and shape heuristics."""
        try:
            if person_crop is None or person_crop.size == 0:
                return "unknown", 0.0
            
            crop_h, crop_w = person_crop.shape[:2]
            if crop_h < 40 or crop_w < 20:
                return "unknown", 0.0
            
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            male_score = 0.0
            female_score = 0.0
            
            # Check for pink/magenta (feminine)
            pink_mask = ((h > 140) & (h < 170) & (s > 50))
            pink_ratio = pink_mask.sum() / h.size
            if pink_ratio > 0.1:
                female_score += 0.3
            
            # Check for red tones
            red_mask = ((h < 10) | (h > 170)) & (s > 50)
            red_ratio = red_mask.sum() / h.size
            if red_ratio > 0.1:
                female_score += 0.15
            
            # Check for very dark clothing (formal wear, more male)
            dark_ratio = (v < 50).sum() / v.size
            if dark_ratio > 0.5:
                male_score += 0.2
            
            # Check for blue tones (slightly more male)
            blue_mask = (h > 100) & (h < 130) & (s > 50)
            blue_ratio = blue_mask.sum() / h.size
            if blue_ratio > 0.15:
                male_score += 0.1
            
            # Body proportions
            aspect = crop_h / crop_w if crop_w > 0 else 0
            if aspect > 2.5:  # Tall and slim
                female_score += 0.15
            elif aspect < 2.0:  # Wider build
                male_score += 0.15
            
            # Determine result
            if male_score > female_score and male_score > 0.1:
                confidence = min(0.7, 0.5 + male_score)
                return "male", confidence
            elif female_score > male_score and female_score > 0.1:
                confidence = min(0.7, 0.5 + female_score)
                return "female", confidence
            
            return "unknown", 0.4
            
        except Exception as e:
            logger.debug(f"Heuristics error: {e}")
            return "unknown", 0.0


class HybridGenderClassifier:
    """
    Hybrid classifier that combines:
    1. PersonAttributesClassifier (body-based, ~85-92% accuracy)
    2. Body Heuristics (color/shape, ~60-70% accuracy)
    
    Uses weighted voting to combine both predictions.
    """
    
    def __init__(self, model_path: str = None):
        self.pa_classifier = PersonAttributesClassifier(model_path)
        self.heuristics = BodyHeuristicsClassifier()
        self.stats = {"model": 0, "heuristic": 0, "combined": 0, "unknown": 0}
        
        if self.pa_classifier.initialized:
            logger.info(f"Gender classifier using {self.pa_classifier.backend} backend with voting")
        else:
            logger.warning("Gender classifier using fallback heuristics only")
    
    def predict(self, frame: np.ndarray, bbox) -> Tuple[str, str, float]:
        """
        Predict gender from detection using weighted voting.
        
        Args:
            frame: Full frame (BGR)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            (gender, method, confidence)
        """
        # Extract person crop
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Add small padding
        pad = 5
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return "unknown", "none", 0.0
        
        # Check minimum size
        crop_h, crop_w = person_crop.shape[:2]
        if crop_h < 40 or crop_w < 20:
            self.stats["unknown"] += 1
            return "unknown", "size", 0.0
        
        # Get PA Model prediction
        pa_gender, pa_conf = None, 0.0
        if self.pa_classifier.initialized:
            attrs = self.pa_classifier.predict_raw(person_crop)
            if attrs is not None:
                is_male_score = float(attrs[0])
                if is_male_score > 0.5:
                    pa_gender = "male"
                    pa_conf = is_male_score
                else:
                    pa_gender = "female"
                    pa_conf = 1.0 - is_male_score
        
        # Get Heuristics prediction
        hr_gender, hr_conf = self.heuristics.predict(person_crop)
        
        # Combine with weighted voting
        gender, confidence, method = self._combine_votes(
            pa_gender, pa_conf,
            hr_gender, hr_conf
        )
        
        # Update stats
        if method == "model":
            self.stats["model"] += 1
        elif method == "heuristic":
            self.stats["heuristic"] += 1
        elif method == "combined":
            self.stats["combined"] += 1
        else:
            self.stats["unknown"] += 1
        
        return gender, method, confidence
    
    def _combine_votes(self, pa_gender, pa_conf, hr_gender, hr_conf) -> Tuple[str, float, str]:
        """Combine PA Model and Heuristics with weighted voting."""
        
        # If PA Model has high confidence, trust it
        if pa_gender and pa_conf > 0.7:
            return pa_gender, pa_conf, "model"
        
        # If PA Model has medium confidence
        if pa_gender and pa_conf > 0.55:
            # Check if heuristics agrees
            if hr_gender == pa_gender:
                # Both agree - boost confidence
                combined_conf = min(0.95, pa_conf + 0.1)
                return pa_gender, combined_conf, "combined"
            else:
                # Disagree - trust PA Model but lower confidence
                return pa_gender, pa_conf * 0.9, "model"
        
        # If PA Model has low confidence (0.5-0.55)
        if pa_gender and pa_conf >= 0.5:
            if hr_gender and hr_gender != "unknown":
                if hr_gender == pa_gender:
                    return pa_gender, max(pa_conf, hr_conf), "combined"
                else:
                    # Conflict - use heuristics if more confident
                    if hr_conf > pa_conf:
                        return hr_gender, hr_conf, "heuristic"
                    else:
                        return pa_gender, pa_conf, "model"
            return pa_gender, pa_conf, "model"
        
        # PA Model failed, use heuristics
        if hr_gender and hr_gender != "unknown":
            return hr_gender, hr_conf, "heuristic"
        
        return "unknown", 0.0, "none"
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        total = sum(self.stats.values())
        if total == 0:
            return {"model_pct": 0, "heuristic_pct": 0, "combined_pct": 0, "unknown_pct": 0, "total": 0}
        
        return {
            "model_pct": round(self.stats["model"] / total * 100, 1),
            "heuristic_pct": round(self.stats["heuristic"] / total * 100, 1),
            "combined_pct": round(self.stats["combined"] / total * 100, 1),
            "unknown_pct": round(self.stats["unknown"] / total * 100, 1),
            "total": total
        }
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.stats = {"model": 0, "heuristic": 0, "combined": 0, "unknown": 0}


# Factory function
def create_gender_classifier(model_path: str = None) -> HybridGenderClassifier:
    """Create a gender classifier instance."""
    return HybridGenderClassifier(model_path)


# Test function
if __name__ == "__main__":
    import sys
    
    print("Person Attributes Classifier Test")
    print(f"ONNX Runtime available: {ONNX_AVAILABLE}")
    print(f"OpenVINO available: {OPENVINO_AVAILABLE}")
    
    # Try to create classifier
    classifier = PersonAttributesClassifier()
    
    if classifier.initialized:
        print(f"Model loaded using: {classifier.backend}")
        
        # Test with sample image if provided
        if len(sys.argv) > 1:
            img = cv2.imread(sys.argv[1])
            if img is not None:
                gender, conf = classifier.predict_gender(img)
                print(f"Gender: {gender} (confidence: {conf:.2f})")
                
                attrs = classifier.predict_all_attributes(img)
                if attrs:
                    print("All attributes:")
                    for name, prob in attrs.items():
                        print(f"  {name}: {prob:.2f}")
    else:
        print("Model not found. Please download the model first.")
        print("\nDownload instructions:")
        print("1. Install OpenVINO: pip install openvino-dev")
        print("2. Download model: omz_downloader --name person-attributes-recognition-crossroad-0234")
        print("3. Convert to ONNX or use .xml/.bin directly")