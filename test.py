"""
System Verification & Diagnostic Test Script.
Runs automated checks on Camera, MediaPipe Hand Tracking, and Model loading.
"""
import os
import sys
import time
import cv2

print("\n" + "=" * 60)
print("     SIGN LANGUAGE SYSTEM - DIAGNOSTIC CHECK")
print("=" * 60)

# Check 1: Python environment
print(f"[1/4] Python Version: {sys.version.split()[0]}")

# Check 2: Model File
from recognize import MODEL_FILE, SignRecognizer
if os.path.exists(MODEL_FILE):
    recog = SignRecognizer()
    signs = ", ".join(recog.class_names)
    print(f"[2/4] Machine Learning Model: FOUND (Classes: {signs})")
else:
    print(f"[2/4] Machine Learning Model: NOT FOUND at {MODEL_FILE}")
    print("      -> Run: python train_model.py")

# Check 3: Hand Detector
try:
    from cvzone.HandTrackingModule import HandDetector
    detector = HandDetector(maxHands=1)
    print("[3/4] MediaPipe Hand Detector: READY")
except Exception as e:
    print(f"[3/4] Hand Detector ERROR: {e}")

# Check 4: Camera
from utils.landmark_utils import open_camera
cap = open_camera(0)
if cap is not None and cap.isOpened():
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"[4/4] Webcam: CONNECTED ({frame.shape[1]}x{frame.shape[0]} resolution)")
    else:
        print("[4/4] Webcam: OPENED but frame read failed")
    cap.release()
else:
    print("[4/4] Webcam: NOT DETECTED (Check camera connection/index)")

print("=" * 60)
print("Diagnostic complete! You are ready to run:")
print("  python app.py")
print("=" * 60 + "\n")
