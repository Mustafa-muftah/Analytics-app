"""
Multi-Classifier Gender Detection Test
PA Model + Body Heuristics (No InsightFace)
"""

import cv2
import sys
import numpy as np

# === LOAD CLASSIFIERS ===

print("=" * 60)
print("Loading classifiers...")
print("=" * 60)

# 1. Person Attributes Model
PA_MODEL_OK = False
pa_classifier = None
try:
    from person_attributes_classifier import PersonAttributesClassifier
    pa_classifier = PersonAttributesClassifier("models/person_attributes.xml")
    PA_MODEL_OK = pa_classifier.initialized
    if PA_MODEL_OK:
        print(f"✓ Person Attributes Model loaded ({pa_classifier.backend})")
    else:
        print("✗ Person Attributes Model failed to load")
except Exception as e:
    print(f"✗ Person Attributes Model error: {e}")

# 2. Body Heuristics (always available)
print("✓ Body Heuristics ready")

# 3. YOLO
from ultralytics import YOLO
yolo = YOLO('yolov8n.pt')
print("✓ YOLO loaded")


def get_pa_model_gender(pa_classifier, person_crop):
    """Get gender from Person Attributes model."""
    if pa_classifier is None or not pa_classifier.initialized:
        return None, 0.0
    
    try:
        attrs = pa_classifier.predict_raw(person_crop)
        if attrs is not None:
            is_male_score = float(attrs[0])
            
            # is_male_score > 0.5 means MALE
            if is_male_score > 0.5:
                return "male", is_male_score
            else:
                return "female", 1.0 - is_male_score
    except:
        pass
    
    return None, 0.0


def get_heuristic_gender(person_crop):
    """Simple body heuristics based on color and shape."""
    try:
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
        
        # Check for very dark clothing
        dark_ratio = (v < 50).sum() / v.size
        if dark_ratio > 0.5:
            male_score += 0.2
        
        # Check for blue tones (slightly more male)
        blue_mask = (h > 100) & (h < 130) & (s > 50)
        blue_ratio = blue_mask.sum() / h.size
        if blue_ratio > 0.15:
            male_score += 0.1
        
        # Check body proportions
        crop_h, crop_w = person_crop.shape[:2]
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
        
    except:
        pass
    
    return None, 0.0


def combined_gender_vote(pa_result, heuristic_result):
    """Combine PA Model and Heuristics with weighted voting."""
    
    pa_gender, pa_conf = pa_result
    hr_gender, hr_conf = heuristic_result
    
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


# === MAIN TEST ===

video_path = sys.argv[1] if len(sys.argv) > 1 else "test3.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Cannot open: {video_path}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Cannot read frame")
    sys.exit(1)

# Detect people
detections = yolo(frame, classes=[0], verbose=False)

print(f"\n{'='*70}")
print(f"Video: {video_path}")
print(f"Frame: {total_frames // 2} | People detected: {len(detections[0].boxes)}")
print(f"{'='*70}\n")

print(f"{'#':<4} {'PA Model':<18} {'Heuristic':<18} {'FINAL RESULT':<20}")
print("-" * 70)

male_count = 0
female_count = 0

for i, box in enumerate(detections[0].boxes):
    bbox = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, bbox)
    
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        continue
    
    # Get predictions
    pa_result = get_pa_model_gender(pa_classifier, person_crop)
    heuristic_result = get_heuristic_gender(person_crop)
    
    # Combined vote
    final_gender, final_conf, final_method = combined_gender_vote(pa_result, heuristic_result)
    
    # Format output
    def fmt(result):
        if result[0] is None:
            return "---"
        return f"{result[0].upper()} ({result[1]:.2f})"
    
    pa_str = fmt(pa_result)
    hr_str = fmt(heuristic_result)
    final_str = f"{final_gender.upper()} ({final_conf:.2f}) [{final_method}]"
    
    print(f"{i+1:<4} {pa_str:<18} {hr_str:<18} {final_str:<20}")
    
    if final_gender == "male":
        male_count += 1
    elif final_gender == "female":
        female_count += 1

print("-" * 70)
print(f"\n>>> FINAL COUNT: {male_count} Male, {female_count} Female")
print(f"{'='*70}")