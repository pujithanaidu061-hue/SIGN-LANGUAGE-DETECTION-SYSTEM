import os
import sys
import time
import pickle
from collections import deque, Counter
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

from utils.landmark_utils import open_camera, extract_features, draw_styled_hud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "sign_language_model.p")


class SignRecognizer:
    def __init__(self, model_path=MODEL_FILE):
        self.model_path = model_path
        self.model_data = None
        self.clf = None
        self.class_names = []
        self.load_model()

        # Stability and debounce tracking
        self.window_size = 10
        self.recent_predictions = deque(maxlen=self.window_size)
        self.confidence_threshold = 0.70
        self.stability_threshold = 7  # 7 out of 10 frames must agree

        self.last_committed_sign = None
        self.last_hand_seen_time = time.time()
        self.sentence = []

    def load_model(self):
        if not os.path.exists(self.model_path):
            return False
        try:
            with open(self.model_path, "rb") as f:
                self.model_data = pickle.load(f)
            self.clf = self.model_data["model"]
            self.class_names = self.model_data["class_names"]
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def process_frame(self, img, hands):
        """
        Takes raw frame and detected hands list from cvzone.
        Returns:
            pred_label (str or None),
            confidence (float),
            stable_sign (str or None)
        """
        if not self.clf:
            return None, 0.0, None

        if not hands:
            # If no hand seen for > 1.2 seconds, reset last_committed_sign so same sign can be repeated
            if time.time() - self.last_hand_seen_time > 1.2:
                self.last_committed_sign = None
            self.recent_predictions.append(None)
            return None, 0.0, None

        self.last_hand_seen_time = time.time()
        hand = hands[0]
        features = extract_features(hand)

        if features is None:
            self.recent_predictions.append(None)
            return None, 0.0, None

        # Predict probabilities
        probs = self.clf.predict_proba([features])[0]
        max_idx = int(np.argmax(probs))
        pred_label = self.class_names[max_idx]
        confidence = float(probs[max_idx])

        # Filter by confidence threshold
        if confidence >= self.confidence_threshold:
            self.recent_predictions.append(pred_label)
        else:
            self.recent_predictions.append(None)

        # Check prediction stability over recent window
        valid_preds = [p for p in self.recent_predictions if p is not None]
        stable_sign = None

        if valid_preds:
            counts = Counter(valid_preds)
            most_common, freq = counts.most_common(1)[0]
            if freq >= self.stability_threshold:
                stable_sign = most_common

        # Debounced sentence building: only add if new sign is stable
        if stable_sign is not None and stable_sign != self.last_committed_sign:
            self.sentence.append(stable_sign)
            self.last_committed_sign = stable_sign
            print(f"[RECOGNIZED] Added '{stable_sign}' to sentence. Current: {' '.join(self.sentence)}")

        return pred_label, confidence, stable_sign

    def clear_sentence(self):
        self.sentence.clear()
        self.last_committed_sign = None

    def backspace(self):
        if self.sentence:
            removed = self.sentence.pop()
            self.last_committed_sign = None
            print(f"[EDIT] Removed '{removed}'. Current sentence: {' '.join(self.sentence)}")


def main():
    print("\n" + "=" * 60)
    print("       REAL-TIME SIGN LANGUAGE RECOGNITION")
    print("=" * 60)

    recognizer = SignRecognizer()
    if not recognizer.clf:
        print("\n[ERROR] Model file not found!")
        print(f"Expected at: {MODEL_FILE}")
        print("Please train the model first by running:")
        print("  python train_model.py\n")
        return

    print(f"Loaded Model:")
    print(f"  Recognizable Signs: {', '.join(recognizer.class_names)}")
    print("\nControls:")
    print("  [C]        : Clear sentence")
    print("  [Backspace]: Delete last word")
    print("  [Q] / [ESC]: Quit")
    print("=" * 60 + "\n")

    cap = open_camera(0)
    if cap is None:
        print("[ERROR] Could not open webcam.")
        return

    detector = HandDetector(maxHands=1)

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("[WARNING] Frame dropped.")
            break

        # Mirror horizontally for natural interaction
        img = cv2.flip(img, 1)

        # Detect hand
        hands, img = detector.findHands(img, draw=False)

        pred_label, confidence, stable_sign = recognizer.process_frame(img, hands)

        # Draw HUD on hand if detected
        if hands:
            hand = hands[0]
            bbox = hand['bbox']
            draw_styled_hud(img, bbox, label=pred_label, confidence=confidence if pred_label else None)

        # Header Bar
        h, w, _ = img.shape
        cv2.rectangle(img, (0, 0), (w, 55), (25, 25, 25), cv2.FILLED)
        cv2.putText(
            img,
            "Sign Language Recognition",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

        # Current recognition indicator
        current_display = pred_label if (pred_label and confidence >= 0.70) else "Detecting..."
        cv2.putText(
            img,
            f"Sign: {current_display}",
            (w - 240, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 128) if pred_label else (160, 160, 160),
            2,
            cv2.LINE_AA,
        )

        # Bottom Sentence Bar
        sentence_str = " ".join(recognizer.sentence)
        if not sentence_str:
            sentence_str = "[Empty sentence - make a sign to begin]"

        cv2.rectangle(img, (0, h - 60), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(img, (10, h - 55), (w - 10, h - 10), (45, 45, 45), cv2.FILLED)

        # Text truncation for long sentences
        display_text = f"Sentence: {sentence_str}"
        if len(display_text) > 48:
            display_text = "..." + display_text[-45:]

        cv2.putText(
            img,
            display_text,
            (25, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Sign Language Recognition System", img)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('c'):
            recognizer.clear_sentence()
        elif key in [ord('b'), 8]:  # 'b' or Backspace
            recognizer.backspace()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Recognition closed.")


if __name__ == "__main__":
    main()


