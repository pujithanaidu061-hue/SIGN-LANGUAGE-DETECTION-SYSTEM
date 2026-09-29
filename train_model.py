import os
import sys
import time
import pickle
import argparse
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils.landmark_utils import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "sign_language_model.p")


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def generate_synthetic_data(samples_per_class=40):
    """
    Generates synthetic landmark data for demonstration when no real images are collected yet.
    Allows beginners to verify the entire training, recognition, and UI pipeline immediately.
    """
    print("\n[DEMO MODE] Generating synthetic baseline landmarks for Hello, Thank you, Yes...")
    classes = ["Hello", "Thank you", "Yes"]
    X, y = [], []

    rng = np.random.RandomState(42)

    for label in classes:
        for _ in range(samples_per_class):
            # Base template for 21 landmarks
            landmarks = []
            wrist = [150.0, 250.0]
            landmarks.append([wrist[0], wrist[1], 0])

            if label == "Hello":  # Open palm (fingers spread upwards)
                for finger_idx in range(5):
                    for joint_idx in range(1, 5):
                        base_x = 100 + finger_idx * 25 + rng.normal(0, 3)
                        base_y = 200 - joint_idx * 35 + rng.normal(0, 3)
                        landmarks.append([base_x, base_y, 0])
            elif label == "Yes":  # Closed fist / nod shape (fingers curled down towards palm)
                for finger_idx in range(5):
                    for joint_idx in range(1, 5):
                        base_x = 120 + finger_idx * 15 + rng.normal(0, 3)
                        base_y = 180 + joint_idx * 10 + rng.normal(0, 3)
                        landmarks.append([base_x, base_y, 0])
            else:  # Thank you (flat fingers tilted or touching chin)
                for finger_idx in range(5):
                    for joint_idx in range(1, 5):
                        base_x = 130 + finger_idx * 20 + joint_idx * 15 + rng.normal(0, 3)
                        base_y = 170 - joint_idx * 20 + rng.normal(0, 3)
                        landmarks.append([base_x, base_y, 0])

            dummy_hand = {
                "lmList": landmarks,
                "bbox": (80, 50, 160, 220)
            }
            feat = extract_features(dummy_hand)
            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y), classes


def extract_dataset_from_images():
    """
    Scans Data/ folder, loads all images, detects hands using MediaPipe/cvzone,
    and extracts normalized 84-dimensional landmark features.
    """
    detector = HandDetector(maxHands=1)
    sign_folders = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(".")
    ]

    X, y = [], []
    stats = {}

    print(f"[INFO] Scanning folders in: {DATA_DIR}")
    for sign in sorted(sign_folders):
        folder_path = os.path.join(DATA_DIR, sign)
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        stats[sign] = {"total_images": len(image_files), "valid_hands": 0}

        print(f"  -> Processing '{sign}' ({len(image_files)} images found)...")

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            hands, _ = detector.findHands(img, draw=False)
            if hands:
                hand = hands[0]
                features = extract_features(hand)
                if features is not None:
                    X.append(features)
                    y.append(sign)
                    stats[sign]["valid_hands"] += 1

    return np.array(X), np.array(y), sorted(sign_folders), stats


def train_and_save(X, y, class_names):
    """
    Trains a Random Forest classifier, evaluates accuracy, and saves model to disk.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\n" + "=" * 60)
    print("                MACHINE LEARNING TRAINING")
    print("=" * 60)
    print(f"Total valid samples: {len(X)}")
    print("Class distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  - {cls:<15}: {count} samples")

    # Check minimum data requirement
    if len(unique_classes) < 2:
        print("\n[ERROR] Training requires at least 2 distinct classes to learn.")
        return False

    min_samples = min(counts)
    can_split = min_samples >= 5 and len(X) >= 20

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n[INFO] Split dataset into {len(X_train)} training and {len(X_test)} test samples.")
    else:
        print("\n[NOTE] Small dataset: using full dataset for training.")
        X_train, y_train = X, y
        X_test, y_test = X, y

    # Train Random Forest Classifier
    # Beginner explanation:
    # A Random Forest consists of 100 decision trees. Each tree votes on the gesture.
    # It is fast, handles non-linear geometric joint boundaries, and avoids overfitting.
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n" + "-" * 40)
    print(f"  Training Accuracy : {train_acc * 100:.1f}%")
    print(f"  Testing Accuracy  : {test_acc * 100:.1f}%")
    print("-" * 40)

    if can_split:
        print("\nDetailed Performance Report:")
        print(classification_report(y_test, test_pred, digits=3))

    # Save trained model and metadata
    ensure_dirs()
    model_payload = {
        "model": clf,
        "class_names": list(unique_classes),
        "feature_count": X.shape[1],
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "timestamp": time.time(),
        "total_samples": len(X),
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_payload, f)

    print(f"\n[SUCCESS] Model successfully saved to:\n  -> {MODEL_FILE}")
    print("=" * 60 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition Model")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo data if real dataset is empty")
    args = parser.parse_args()

    ensure_dirs()

    X, y, class_names, stats = extract_dataset_from_images()

    total_images = sum(s["total_images"] for s in stats.values()) if stats else 0
    total_valid = len(X)

    if total_valid == 0:
        print("\n" + "!" * 60)
        print("  NOTICE: No valid hand training images found in Data/ folder!")
        print("!" * 60)
        print("Your Data/ folders (Hello, Thank you, Yes) currently have 0 images.")
        print("To collect images using your webcam, run:")
        print("  python data_collection.py\n")

        if args.demo:
            X, y, class_names = generate_synthetic_data(samples_per_class=50)
        elif total_images == 0:
            print("To allow you to test real-time recognition and the GUI immediately,")
            print("we can generate an initial baseline model with synthetic landmarks.")
            try:
                response = input("Generate baseline demo model now? [Y/n]: ").strip().lower()
            except EOFError:
                response = "y"
            if response in ["", "y", "yes"]:
                X, y, class_names = generate_synthetic_data(samples_per_class=50)
            else:
                print("Exiting. Please collect images and run train_model.py again.")
                return
        else:
            return

    train_and_save(X, y, class_names)


if __name__ == "__main__":
    main()
