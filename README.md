# Sign Language Recognition System

An AI-powered, real-time Sign Language Recognition and Translation System designed for deaf and non-speaking communication. The system detects hand gestures via webcam, extracts invariant geometric landmark features, classifies them using machine learning, and constructs full sentences with built-in prediction stabilization and debounce control.

---

## 📁 Project Architecture

```
sign language detection/
│
├── Data/                       # Dataset directories (one folder per gesture)
│   ├── Hello/                  # Captured images for 'Hello'
│   ├── Thank you/              # Captured images for 'Thank you'
│   └── Yes/                    # Captured images for 'Yes'
│
├── models/
│   └── sign_language_model.p   # Trained Random Forest classifier & label metadata
│
├── utils/
│   ├── __init__.py
│   └── landmark_utils.py       # Landmark extraction, normalization & safe cropping
│
├── data_collection.py          # Crash-proof interactive image collection tool
├── train_model.py              # Landmark dataset processor & Random Forest trainer
├── recognize.py                # Standalone OpenCV real-time recognition with debounce
├── app.py                      # Modern CustomTkinter GUI application
├── test.py                     # Diagnostic check script
├── camera_test.py              # Quick camera hardware check
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation & usage manual
```

---

## 🚀 Quick Start Guide (Beginner Friendly)

All commands should be run using your project's virtual environment in PowerShell or Command Prompt.

### Step 0: Run Diagnostic Check
Verify your webcam, MediaPipe model, and environment in one command:
```powershell
.\venv\Scripts\python.exe test.py
```
**Expected Output:**
```
[1/4] Python Version: 3.14.0
[2/4] Machine Learning Model: FOUND (Classes: Hello, Thank you, Yes)
[3/4] MediaPipe Hand Detector: READY
[4/4] Webcam: CONNECTED (640x480 resolution)
```

---

### Step 1: Collect Hand Sign Images
Use the interactive data collection tool to capture 30–50 samples per sign.

```powershell
.\venv\Scripts\python.exe data_collection.py
```

**Controls inside the camera window:**
- `[1]`, `[2]`, `[3]`: Switch active sign between `Hello`, `Thank you`, and `Yes`
- `[N]`: Switch to the next sign
- `[S]`: Capture and save 1 image to the active sign folder
- `[B]`: Toggle **Continuous Burst Mode** (automatically saves a frame every 200ms while you hold the gesture)
- `[Q]` or `[ESC]`: Quit data collection

> **Crash-Proof Protection**: The cropping engine uses boundary clamping so moving your hand to the edge of the frame will never crash the program.

---

### Step 2: Train the Machine Learning Model
Once you have collected images in `Data/Hello`, `Data/Thank you`, and `Data/Yes`:

```powershell
.\venv\Scripts\python.exe train_model.py
```

**What this step does:**
1. Loads all images in `Data/` folders.
2. Extracts 21 hand landmarks (joints) using MediaPipe.
3. Computes 84 invariant geometric features (wrist-relative and bounding-box normalized).
4. Trains a `RandomForestClassifier` (100 decision trees).
5. Evaluates model accuracy on an 80/20 train/test split.
6. Saves the trained model to `models/sign_language_model.p`.

---

### Step 3: Run the Modern GUI Application
Launch the full desktop application for demonstration:

```powershell
.\venv\Scripts\python.exe app.py
```

**Application Features:**
- **Header**: Project title, subtitle, and live status pills for Camera, ML Model, and Hand Tracking.
- **Live Camera Feed**: Smooth webcam stream with hand landmarks and a sleek bounding box HUD.
- **Current Recognition Card**: Displays the active recognized sign in large bold text with a real-time confidence percentage bar.
- **Constructed Sentence Area**: Assembles words into complete sentences (`HELLO THANK YOU YES`).
- **Debounce & Stability**: Prevents duplicate words (e.g. `HELLO HELLO HELLO`) using an 8-frame consensus filter and hold cooldown.
- **Sentence Actions**: "Clear Sentence", "Backspace", and "Copy Text" buttons.
- **System Controls**: "Start Camera", "Stop Camera", and "Data Collector" quick-launch.

---

### Step 4: Standalone OpenCV Recognition (Alternative)
If you want to run recognition directly in a lightweight OpenCV window without the GUI:

```powershell
.\venv\Scripts\python.exe recognize.py
```
- Press `[C]` to clear the sentence.
- Press `[B]` or `[Backspace]` to remove the last word.
- Press `[Q]` to exit.

---

## 🧠 How the Machine Learning Model Works

### Why Hand Landmarks instead of Raw Image Pixels?
- **Raw Pixels**: An image has thousands of pixels that change drastically with lighting, wall color, and skin tone.
- **Hand Landmarks**: MediaPipe identifies 21 3D joint landmarks (wrist, knuckles, fingertips).
- **Normalization**:
  - We calculate every joint coordinate relative to the wrist (landmark 0):
    $$\text{rel\_x}_i = \frac{x_i - x_0}{\text{hand\_size}}, \quad \text{rel\_y}_i = \frac{y_i - y_0}{\text{hand\_size}}$$
  - This results in an **84-dimensional mathematical signature** that captures pure hand shape.
  - The model does not care whether your hand is on the left, right, far away, close up, or in dim light.
- **Random Forest Classifier**:
  - Uses 100 decision trees to classify hand configurations.
  - Predicts in less than 1 millisecond per frame.
  - Provides probability confidence scores for filtering out false positives.

### Sentence Builder & Debounce Mechanism
To prevent a held gesture from spamming the sentence repeatedly:
1. **Confidence Threshold**: Predictions with confidence $< 70\%$ are ignored.
2. **Rolling Window Consensus**: The model maintains a history of the last 10 frames. At least 7 out of 10 frames must agree on the same sign before it is accepted as stable.
3. **State Transition**: A stable sign is added to the sentence once. It will not be added again while held. Lowering the hand for $> 1.2$ seconds resets the state, allowing repeated words when intended.

---

## 🛠️ Adding New Signs

To add a new sign (e.g., `Good`, `Please`, `Help`):
1. Create a new folder: `Data/<NewSignName>`
2. Run `python data_collection.py` and capture 30–50 photos of the new gesture.
3. Run `python train_model.py`.
4. The model will automatically train on all signs and save the new multi-class model!

---

## 📦 Dependencies
- Python 3.10+
- `opencv-python`
- `cvzone`
- `mediapipe`
- `scikit-learn`
- `numpy`
- `customtkinter`
- `pillow`
- `joblib`
