import cv2
import os
import time
import numpy as np

from cvzone.HandTrackingModule import HandDetector
from utils.landmark_utils import open_camera, crop_hand_safe


# ============================================================
# BASE DATASET DIRECTORY
# ============================================================

BASE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Data"
)

DEFAULT_SIGNS = ["Hello", "Thank you", "Yes"]


# ============================================================
# FOLDER FUNCTIONS
# ============================================================

def ensure_folders_exist():
    """Ensure standard Data folders exist."""

    os.makedirs(BASE_DATA_DIR, exist_ok=True)

    for sign in DEFAULT_SIGNS:
        os.makedirs(
            os.path.join(BASE_DATA_DIR, sign),
            exist_ok=True
        )


def get_existing_signs():
    """Returns list of all sign folders in Data directory."""

    if not os.path.exists(BASE_DATA_DIR):
        return DEFAULT_SIGNS

    signs = [
        d
        for d in os.listdir(BASE_DATA_DIR)
        if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
        and not d.startswith(".")
    ]

    return sorted(signs) if signs else DEFAULT_SIGNS


def count_images(sign_name):
    """Counts existing saved samples."""

    folder = os.path.join(BASE_DATA_DIR, sign_name)

    if not os.path.exists(folder):
        return 0

    return len([
        f
        for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])


# ============================================================
# TWO-HAND LANDMARK FEATURE EXTRACTION
# ============================================================

def normalize_landmarks(lm_list):
    """
    Convert 21 hand landmarks into 63 normalized features.

    Each landmark contains:
        x, y, z

    21 landmarks × 3 = 63 features.

    Coordinates are normalized relative to the wrist
    (landmark 0).
    """

    if not lm_list or len(lm_list) < 21:
        return [0.0] * 63

    # Wrist landmark
    wrist_x, wrist_y, wrist_z = lm_list[0]

    features = []

    for x, y, z in lm_list:

        features.extend([
            float(x - wrist_x),
            float(y - wrist_y),
            float(z - wrist_z)
        ])

    return features


def extract_two_hand_features(hands):
    """
    Extract features from BOTH hands.

    Left hand:
        21 landmarks × 3 = 63 features

    Right hand:
        21 landmarks × 3 = 63 features

    Total:
        126 features

    If only one hand is detected, the other hand is
    represented by 63 zeros.
    """

    left_features = [0.0] * 63
    right_features = [0.0] * 63

    for hand in hands:

        lm_list = hand.get("lmList", [])

        features = normalize_landmarks(lm_list)

        hand_type = hand.get("type", "")

        if hand_type == "Left":

            left_features = features

        elif hand_type == "Right":

            right_features = features

    # Left first + Right second
    return left_features + right_features


# ============================================================
# SAVE TWO-HAND DATA
# ============================================================

def save_sample(
    current_sign,
    display_img,
    two_hand_features
):
    """
    Save one training sample.

    Saves:
        1. Camera image
        2. 126 landmark features
    """

    folder = os.path.join(
        BASE_DATA_DIR,
        current_sign
    )

    os.makedirs(folder, exist_ok=True)

    timestamp = int(time.time() * 1000)

    # --------------------------------------------------------
    # Save camera image
    # --------------------------------------------------------

    image_name = f"Image_{timestamp}.jpg"

    image_path = os.path.join(
        folder,
        image_name
    )

    cv2.imwrite(
        image_path,
        display_img
    )

    # --------------------------------------------------------
    # Save landmark features
    # --------------------------------------------------------

    feature_name = f"Features_{timestamp}.npy"

    feature_path = os.path.join(
        folder,
        feature_name
    )

    np.save(
        feature_path,
        np.array(
            two_hand_features,
            dtype=np.float32
        )
    )

    print(
        f"[SAVED] {current_sign}"
    )

    print(
        f"        Image    : {image_name}"
    )

    print(
        f"        Features : {feature_name}"
    )

    print(
        f"        Features : {len(two_hand_features)}"
    )

    return image_name


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    ensure_folders_exist()

    signs = get_existing_signs()

    current_sign_index = 0

    # --------------------------------------------------------
    # Console information
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "             SIGN LANGUAGE DATA COLLECTION TOOL"
    )

    print("=" * 70)

    print(
        "Available Signs:"
    )

    for idx, sign in enumerate(signs):

        cnt = count_images(sign)

        print(
            f"  [{idx + 1}] "
            f"{sign:<15} "
            f"(Currently: {cnt} images)"
        )

    print("\nKeyboard Controls inside Camera Window:")

    print(
        "  [1-9]      : Switch sign directly"
    )

    print(
        "  [N]        : Next sign"
    )

    print(
        "  [S]        : Save single two-hand sample"
    )

    print(
        "  [B]        : Toggle continuous burst mode"
    )

    print(
        "  [Q] / [ESC]: Quit"
    )

    print("=" * 70)

    print(
        "\nTwo-hand feature format:"
    )

    print(
        "  Left Hand  = 63 features"
    )

    print(
        "  Right Hand = 63 features"
    )

    print(
        "  Total      = 126 features"
    )

    print("=" * 70 + "\n")


    # ========================================================
    # OPEN CAMERA
    # ========================================================

    cap = open_camera(0)

    if cap is None:

        print(
            "[ERROR] Could not open camera."
        )

        print(
            "Please check webcam connection."
        )

        return


    # ========================================================
    # TWO-HAND DETECTOR
    # ========================================================

    detector = HandDetector(
        maxHands=2
    )

    img_size = 300
    offset = 20


    # ========================================================
    # BURST SETTINGS
    # ========================================================

    burst_mode = False

    last_burst_time = 0

    burst_interval = 0.2


    # ========================================================
    # SAVE FEEDBACK
    # ========================================================

    save_feedback_text = ""

    save_feedback_timer = 0


    # ========================================================
    # MAIN CAMERA LOOP
    # ========================================================

    while True:

        success, img = cap.read()

        if not success or img is None:

            print(
                "[WARNING] Frame dropped or camera disconnected."
            )

            break


        # ----------------------------------------------------
        # Flip camera horizontally
        # ----------------------------------------------------

        img = cv2.flip(
            img,
            1
        )

        display_img = img.copy()


        # ----------------------------------------------------
        # Current sign
        # ----------------------------------------------------

        current_sign = signs[
            current_sign_index
        ]

        current_count = count_images(
            current_sign
        )


        # ====================================================
        # DETECT UP TO TWO HANDS
        # ====================================================

        hands, img_drawn = detector.findHands(
            img,
            draw=False
        )


        # ----------------------------------------------------
        # Number of detected hands
        # ----------------------------------------------------

        number_of_hands = len(hands)


        # ====================================================
        # EXTRACT TWO-HAND FEATURES
        # ====================================================

        if hands:

            two_hand_features = (
                extract_two_hand_features(hands)
            )

        else:

            two_hand_features = [0.0] * 126


        # ====================================================
        # VARIABLES FOR HAND CROPS
        # ====================================================

        left_crop = None

        right_crop = None


        # ====================================================
        # PROCESS EACH HAND
        # ====================================================

        if hands:

            for hand in hands:

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                bbox = hand["bbox"]

                x, y, w, h = bbox


                # ------------------------------------------------
                # Left / Right
                # ------------------------------------------------

                hand_type = hand.get(
                    "type",
                    "Unknown"
                )


                # ------------------------------------------------
                # Safe crop
                # ------------------------------------------------

                img_crop, img_white = crop_hand_safe(
                    img,
                    bbox,
                    offset=offset,
                    img_size=img_size
                )


                # =================================================
                # DRAW BOUNDING BOX
                # =================================================

                cv2.rectangle(
                    display_img,

                    (
                        x - offset,
                        y - offset
                    ),

                    (
                        x + w + offset,
                        y + h + offset
                    ),

                    (0, 255, 128),

                    2
                )


                # =================================================
                # DISPLAY HAND TYPE
                # =================================================

                cv2.putText(
                    display_img,

                    hand_type,

                    (
                        x,
                        max(
                            90,
                            y - offset - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (0, 255, 128),

                    2,

                    cv2.LINE_AA
                )


                # =================================================
                # STORE HAND CROP
                # =================================================

                if img_white is not None:

                    if hand_type == "Left":

                        left_crop = img_white

                    elif hand_type == "Right":

                        right_crop = img_white


        # ====================================================
        # DISPLAY HAND CROPS
        # ====================================================

        if left_crop is not None:

            cv2.imshow(
                "Left Hand",
                left_crop
            )

        if right_crop is not None:

            cv2.imshow(
                "Right Hand",
                right_crop
            )


        # ====================================================
        # HUD
        # ====================================================

        hud_bg = (
            30,
            30,
            30
        )

        cv2.rectangle(
            display_img,

            (
                0,
                0
            ),

            (
                display_img.shape[1],
                100
            ),

            hud_bg,

            cv2.FILLED
        )


        # ----------------------------------------------------
        # Active sign
        # ----------------------------------------------------

        cv2.putText(
            display_img,

            f"Active Sign: {current_sign.upper()}",

            (
                15,
                30
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 215, 255),

            2,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Hand count
        # ----------------------------------------------------

        cv2.putText(
            display_img,

            f"Hands Detected: {number_of_hands}/2",

            (
                15,
                60
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (200, 200, 200),

            1,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Image count / burst
        # ----------------------------------------------------

        cv2.putText(
            display_img,

            (
                f"Images: {current_count} | "
                f"Burst: "
                f"{'ON [Active]' if burst_mode else 'OFF'}"
            ),

            (
                15,
                85
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (200, 200, 200),

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # FOOTER
        # ====================================================

        footer_y = (
            display_img.shape[0] - 15
        )

        cv2.rectangle(
            display_img,

            (
                0,
                display_img.shape[0] - 35
            ),

            (
                display_img.shape[1],
                display_img.shape[0]
            ),

            (
                20,
                20,
                20
            ),

            cv2.FILLED
        )


        cv2.putText(
            display_img,

            (
                "Keys: [S] Save | "
                "[B] Burst | "
                "[1-9] Select Sign | "
                "[N] Next Sign | "
                "[Q] Quit"
            ),

            (
                15,
                footer_y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (180, 180, 180),

            1,

            cv2.LINE_AA
        )


        # ====================================================
        # SAVE FEEDBACK
        # ====================================================

        if (
            time.time() - save_feedback_timer
            < 0.6
        ):

            cv2.rectangle(
                display_img,

                (
                    display_img.shape[1] - 220,
                    15
                ),

                (
                    display_img.shape[1] - 15,
                    55
                ),

                (
                    0,
                    180,
                    0
                ),

                cv2.FILLED
            )


            cv2.putText(
                display_img,

                save_feedback_text,

                (
                    display_img.shape[1] - 210,
                    42
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2,

                cv2.LINE_AA
            )


        # ====================================================
        # BURST CAPTURE
        # ====================================================

        if (
            burst_mode
            and hands
            and time.time() - last_burst_time
            >= burst_interval
        ):

            # ------------------------------------------------
            # Save only when at least one hand is detected
            # ------------------------------------------------

            save_sample(
                current_sign,
                display_img,
                two_hand_features
            )


            # ------------------------------------------------
            # Update burst timer
            # ------------------------------------------------

            last_burst_time = time.time()


            # ------------------------------------------------
            # Update feedback
            # ------------------------------------------------

            save_feedback_text = (
                f"SAVED #{current_count + 1}"
            )

            save_feedback_timer = (
                time.time()
            )


        # ====================================================
        # SHOW MAIN CAMERA
        # ====================================================

        cv2.imshow(
            "Sign Language Data Collector",
            display_img
        )


        # ====================================================
        # KEYBOARD INPUT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key in [
            ord("q"),
            27
        ]:

            break


        # ====================================================
        # MANUAL SAVE
        # ====================================================

        elif key == ord("s"):

            if hands:

                saved_file = save_sample(
                    current_sign,
                    display_img,
                    two_hand_features
                )


                save_feedback_text = (
                    f"SAVED #{current_count + 1}"
                )

                save_feedback_timer = (
                    time.time()
                )

            else:

                print(
                    "[WARNING] "
                    "No hand detected in frame to save!"
                )


        # ====================================================
        # BURST MODE
        # ====================================================

        elif key == ord("b"):

            burst_mode = not burst_mode

            print(
                "[BURST MODE] "
                f"{'ENABLED' if burst_mode else 'DISABLED'}"
            )


            # Reset timer when burst starts
            if burst_mode:

                last_burst_time = (
                    time.time()
                )


        # ====================================================
        # NEXT SIGN
        # ====================================================

        elif key == ord("n"):

            current_sign_index = (
                current_sign_index + 1
            ) % len(signs)


            print(
                "[SWITCH] Switched to: "
                f"{signs[current_sign_index]}"
            )


        # ====================================================
        # DIRECT SIGN SELECTION
        # ====================================================

        elif (
            ord("1")
            <= key
            <= ord("9")
        ):

            idx = (
                key - ord("1")
            )


            if idx < len(signs):

                current_sign_index = idx

                print(
                    "[SWITCH] Switched to: "
                    f"{signs[current_sign_index]}"
                )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print(
        "[INFO] Data collection stopped."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
