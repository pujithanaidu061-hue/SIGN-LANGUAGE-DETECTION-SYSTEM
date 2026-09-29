import os
import sys
import time
import threading
import subprocess
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
import cv2
from cvzone.HandTrackingModule import HandDetector

from recognize import SignRecognizer
from utils.landmark_utils import open_camera, draw_styled_hud

# Set modern look and feel
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language Recognition System")
        self.geometry("1180x760")
        self.minsize(980, 680)

        # Initialize core components
        self.recognizer = SignRecognizer()
        self.detector = HandDetector(maxHands=1)

        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.camera_thread = None

        self.current_sign_text = "None"
        self.current_confidence = 0.0
        self.hand_detected = False

        self._build_ui()
        self._update_status_indicators()

        # Handle window close cleanly
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _build_ui(self):
        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=3)  # Camera column
        self.grid_columnconfigure(1, weight=2)  # Controls/Sentence column

        # -------------------------------------------------------------
        # 1. HEADER
        # -------------------------------------------------------------
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#18181b")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)

        title_label = ctk.CTkLabel(
            header_frame,
            text="Sign Language Recognition System",
            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color="#38bdf8",
        )
        title_label.pack(anchor="w", padx=25, pady=(12, 2))

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Real-Time Hand Gesture to Text Translation for Assistive Communication",
            font=ctk.CTkFont(family="Helvetica", size=13),
            text_color="#94a3b8",
        )
        subtitle_label.pack(anchor="w", padx=25, pady=(0, 10))

        # Status indicator row inside header
        self.status_bar = ctk.CTkFrame(header_frame, fg_color="transparent")
        self.status_bar.pack(anchor="w", padx=25, pady=(0, 10))

        self.status_camera = ctk.CTkLabel(
            self.status_bar,
            text="● Camera: Disconnected",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
        )
        self.status_camera.pack(side="left", padx=(0, 20))

        self.status_model = ctk.CTkLabel(
            self.status_bar,
            text="● Model: Checking...",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#facc15",
        )
        self.status_model.pack(side="left", padx=(0, 20))

        self.status_hand = ctk.CTkLabel(
            self.status_bar,
            text="● Hand: Searching",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94a3b8",
        )
        self.status_hand.pack(side="left", padx=(0, 20))

        # -------------------------------------------------------------
        # 2. LEFT PANEL: LIVE CAMERA AREA
        # -------------------------------------------------------------
        left_panel = ctk.CTkFrame(self, corner_radius=12, fg_color="#27272a")
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=15)
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        cam_title_bar = ctk.CTkFrame(left_panel, fg_color="transparent")
        cam_title_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 5))

        ctk.CTkLabel(
            cam_title_bar,
            text="LIVE CAMERA FEED",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e2e8f0",
        ).pack(side="left")

        # Video canvas / preview label
        self.video_label = ctk.CTkLabel(
            left_panel,
            text="Camera is OFF\n\nClick 'Start Camera' to begin recognition.",
            font=ctk.CTkFont(size=16),
            text_color="#71717a",
            fg_color="#18181b",
            corner_radius=8,
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 15))

        # -------------------------------------------------------------
        # 3. RIGHT PANEL: RECOGNITION, SENTENCE & CONTROLS
        # -------------------------------------------------------------
        right_panel = ctk.CTkFrame(self, corner_radius=12, fg_color="#27272a")
        right_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=15)

        # Card A: Current Recognized Sign
        recog_card = ctk.CTkFrame(right_panel, corner_radius=10, fg_color="#18181b")
        recog_card.pack(fill="x", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            recog_card,
            text="CURRENT RECOGNITION",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94a3b8",
        ).pack(anchor="w", padx=15, pady=(10, 2))

        self.sign_display = ctk.CTkLabel(
            recog_card,
            text="---",
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold"),
            text_color="#38bdf8",
        )
        self.sign_display.pack(anchor="w", padx=15, pady=(0, 5))

        # Confidence bar & percentage
        conf_row = ctk.CTkFrame(recog_card, fg_color="transparent")
        conf_row.pack(fill="x", padx=15, pady=(0, 12))

        self.confidence_bar = ctk.CTkProgressBar(conf_row, height=10, fg_color="#3f3f46", progress_color="#38bdf8")
        self.confidence_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.confidence_bar.set(0)

        self.confidence_label = ctk.CTkLabel(
            conf_row,
            text="0%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94a3b8",
            width=45,
        )
        self.confidence_label.pack(side="right")

        # Card B: Constructed Sentence Area
        sentence_card = ctk.CTkFrame(right_panel, corner_radius=10, fg_color="#18181b")
        sentence_card.pack(fill="both", expand=True, padx=15, pady=5)

        ctk.CTkLabel(
            sentence_card,
            text="CONSTRUCTED SENTENCE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94a3b8",
        ).pack(anchor="w", padx=15, pady=(10, 5))

        self.sentence_box = ctk.CTkTextbox(
            sentence_card,
            font=ctk.CTkFont(family="Consolas", size=16),
            wrap="word",
            fg_color="#09090b",
            border_width=1,
            border_color="#3f3f46",
            text_color="#f1f5f9",
        )
        self.sentence_box.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # Sentence action buttons
        btn_row = ctk.CTkFrame(sentence_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=(0, 12))

        self.clear_btn = ctk.CTkButton(
            btn_row,
            text="Clear Sentence",
            command=self.clear_sentence,
            fg_color="#ef4444",
            hover_color="#dc2626",
            height=32,
        )
        self.clear_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.backspace_btn = ctk.CTkButton(
            btn_row,
            text="Backspace",
            command=self.backspace_sentence,
            fg_color="#475569",
            hover_color="#334155",
            height=32,
        )
        self.backspace_btn.pack(side="left", fill="x", expand=True, padx=5)

        self.copy_btn = ctk.CTkButton(
            btn_row,
            text="Copy Text",
            command=self.copy_sentence,
            fg_color="#0284c7",
            hover_color="#0369a1",
            height=32,
        )
        self.copy_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Card C: Main Controls
        controls_card = ctk.CTkFrame(right_panel, corner_radius=10, fg_color="#18181b")
        controls_card.pack(fill="x", padx=15, pady=(10, 15))

        ctk.CTkLabel(
            controls_card,
            text="CONTROLS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#94a3b8",
        ).pack(anchor="w", padx=15, pady=(10, 6))

        ctrl_btn_grid = ctk.CTkFrame(controls_card, fg_color="transparent")
        ctrl_btn_grid.pack(fill="x", padx=15, pady=(0, 12))
        ctrl_btn_grid.grid_columnconfigure((0, 1), weight=1)

        self.start_cam_btn = ctk.CTkButton(
            ctrl_btn_grid,
            text="▶ Start Camera",
            command=self.start_camera,
            fg_color="#10b981",
            hover_color="#059669",
            height=38,
            font=ctk.CTkFont(weight="bold"),
        )
        self.start_cam_btn.grid(row=0, column=0, padx=(0, 5), pady=4, sticky="ew")

        self.stop_cam_btn = ctk.CTkButton(
            ctrl_btn_grid,
            text="⏹ Stop Camera",
            command=self.stop_camera,
            fg_color="#64748b",
            hover_color="#475569",
            height=38,
            state="disabled",
        )
        self.stop_cam_btn.grid(row=0, column=1, padx=(5, 0), pady=4, sticky="ew")

        self.collector_btn = ctk.CTkButton(
            ctrl_btn_grid,
            text="📷 Data Collector",
            command=self.launch_data_collector,
            fg_color="#334155",
            hover_color="#1e293b",
            height=32,
        )
        self.collector_btn.grid(row=1, column=0, padx=(0, 5), pady=4, sticky="ew")

        self.exit_btn = ctk.CTkButton(
            ctrl_btn_grid,
            text="✕ Exit",
            command=self.on_exit,
            fg_color="#3f3f46",
            hover_color="#27272a",
            height=32,
        )
        self.exit_btn.grid(row=1, column=1, padx=(5, 0), pady=4, sticky="ew")

    def _update_status_indicators(self):
        # Update Model status
        if self.recognizer.clf is not None:
            classes_str = ", ".join(self.recognizer.class_names)
            self.status_model.configure(
                text=f"● Model: Loaded ({classes_str})",
                text_color="#4ade80",
            )
        else:
            self.status_model.configure(
                text="● Model: Not Loaded (Run train_model.py)",
                text_color="#f87171",
            )

        # Update Camera status
        if self.is_camera_running:
            self.status_camera.configure(text="● Camera: Connected", text_color="#4ade80")
            self.start_cam_btn.configure(state="disabled")
            self.stop_cam_btn.configure(state="normal")
        else:
            self.status_camera.configure(text="● Camera: Disconnected", text_color="#f87171")
            self.start_cam_btn.configure(state="normal")
            self.stop_cam_btn.configure(state="disabled")

        # Update Hand status
        if self.is_camera_running and self.hand_detected:
            self.status_hand.configure(text="● Hand: Active", text_color="#4ade80")
        elif self.is_camera_running:
            self.status_hand.configure(text="● Hand: Searching", text_color="#facc15")
        else:
            self.status_hand.configure(text="● Hand: Idle", text_color="#94a3b8")

    def start_camera(self):
        if self.is_camera_running:
            return

        self.cap = open_camera(0)
        if self.cap is None:
            messagebox.showerror("Camera Error", "Could not open webcam. Check connection or index.")
            return

        self.is_camera_running = True
        self._update_status_indicators()

        self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.camera_thread.start()

        # Begin UI render loop
        self.after(20, self._render_loop)

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.configure(
            image="",
            text="Camera is stopped.\n\nClick 'Start Camera' to resume recognition.",
        )
        self.hand_detected = False
        self.sign_display.configure(text="---", text_color="#38bdf8")
        self.confidence_bar.set(0)
        self.confidence_label.configure(text="0%")
        self._update_status_indicators()

    def _camera_worker(self):
        """Dedicated background thread for webcam capturing and hand processing."""
        while self.is_camera_running and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)

            # Detect hand
            hands, _ = self.detector.findHands(frame, draw=False)
            pred_label, confidence, stable_sign = self.recognizer.process_frame(frame, hands)

            # Draw HUD
            if hands:
                self.hand_detected = True
                bbox = hands[0]['bbox']
                draw_styled_hud(frame, bbox, label=pred_label, confidence=confidence if pred_label else None)
            else:
                self.hand_detected = False

            # Update shared frame safely
            with self.frame_lock:
                self.current_frame = frame
                self.current_sign_text = pred_label if pred_label and confidence >= 0.70 else "..."
                self.current_confidence = confidence if pred_label else 0.0

            time.sleep(0.01)

    def _render_loop(self):
        """Tkinter main thread loop for rendering frames and updating sentence."""
        if not self.is_camera_running:
            return

        frame_to_show = None
        current_sign = "---"
        confidence = 0.0

        with self.frame_lock:
            if self.current_frame is not None:
                frame_to_show = self.current_frame.copy()
            current_sign = self.current_sign_text
            confidence = self.current_confidence

        if frame_to_show is not None:
            # Resize frame to fit UI video area smoothly
            target_w = 640
            target_h = 480
            frame_resized = cv2.resize(frame_to_show, (target_w, target_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(frame_rgb)
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(target_w, target_h))

            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk  # keep reference

        # Update recognition card
        if current_sign != "...":
            self.sign_display.configure(text=current_sign.upper(), text_color="#34d399")
        else:
            self.sign_display.configure(text="DETECTING...", text_color="#94a3b8")

        self.confidence_bar.set(confidence)
        self.confidence_label.configure(text=f"{int(confidence * 100)}%")

        # Update sentence text box
        sentence_str = " ".join(self.recognizer.sentence)
        current_box_text = self.sentence_box.get("1.0", "end-1c")
        if current_box_text != sentence_str:
            self.sentence_box.delete("1.0", "end")
            self.sentence_box.insert("1.0", sentence_str)
            self.sentence_box.see("end")

        self._update_status_indicators()

        # Schedule next render
        self.after(25, self._render_loop)

    def clear_sentence(self):
        self.recognizer.clear_sentence()
        self.sentence_box.delete("1.0", "end")

    def backspace_sentence(self):
        self.recognizer.backspace()
        self.sentence_box.delete("1.0", "end")
        self.sentence_box.insert("1.0", " ".join(self.recognizer.sentence))

    def copy_sentence(self):
        text = " ".join(self.recognizer.sentence)
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            messagebox.showinfo("Copied", "Sentence copied to clipboard!")

    def launch_data_collector(self):
        was_running = self.is_camera_running
        if was_running:
            self.stop_camera()
        script_path = os.path.join(BASE_DIR, "data_collection.py")
        subprocess.Popen([sys.executable, script_path])

    def on_exit(self):
        self.stop_camera()
        self.destroy()


def main():
    app = SignLanguageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
