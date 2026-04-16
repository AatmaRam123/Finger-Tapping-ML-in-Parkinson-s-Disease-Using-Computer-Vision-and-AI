# This file is responsible for detecting hand landmarks using MediaPipe,
# tracking the thumb and index finger, and estimating finger tapping behaviour
# from a webcam or video file.

import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque

# -------- SETTINGS --------
VIDEO_SOURCE = r"D:\PDA\PDAV\C30.MOV"   # 0 = webcam
# To test a saved video, replace with:
# VIDEO_SOURCE = r".\PDAV\PD2_LEFT.MOV"

SMOOTH_WINDOW = 5
ANALYSIS_WINDOW = 120
WINDOW_NAME = "MediaPipe Finger Tapping Demo"
# --------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

THUMB_TIP = 4
INDEX_TIP = 8

distance_history = deque(maxlen=SMOOTH_WINDOW)
signal_window = deque(maxlen=ANALYSIS_WINDOW)

tap_count = 0
prev_state = "OPEN"

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1000, 700)


def estimate_score(signal):
    # Simple UPDRS-style estimate
    if len(signal) < 20:
        return None

    x = np.array(signal, dtype=float)

    amplitude = np.max(x) - np.min(x)
    std = np.std(x)
    speed = np.mean(np.abs(np.diff(x))) if len(x) > 1 else 0.0

    score = 0

    if amplitude < 200:
        score += 1
    if std > 80:
        score += 1
    if speed < 12:
        score += 1

    return min(4, score)


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished video or failed to read frame.")
            break

        frame = cv2.resize(frame, (1000, 700))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            thumb = hand_landmarks.landmark[THUMB_TIP]
            indexf = hand_landmarks.landmark[INDEX_TIP]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(indexf.x * w), int(indexf.y * h)

            cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)
            cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
            cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 0), 2)

            dist = math.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)

            distance_history.append(dist)
            smooth = float(np.mean(distance_history))
            signal_window.append(smooth)

            state = "CLOSE" if smooth < 120 else "OPEN"
            if prev_state == "CLOSE" and state == "OPEN":
                tap_count += 1
            prev_state = state

            score = estimate_score(signal_window)

            cv2.putText(frame, f"Distance: {smooth:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Taps: {tap_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            if score is not None:
                cv2.putText(frame, f"Estimated Score: {score}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, "Thumb tip (green), Index tip (red)", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)

        else:
            cv2.putText(frame, "No hand detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "Press Q to quit", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()