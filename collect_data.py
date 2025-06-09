import cv2
import os
import mediapipe as mp
import numpy as np
import math

# Constants
DATA_DIR = "C:/Users/oktay/AILab/Project/gesture_data"
LABELS = ["up", "down", "left", "right"]
SAMPLES_PER_LABEL = 200

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def get_direction_from_angle(angle_degrees):
    angle_degrees = (angle_degrees + 360) % 360
    if 315 <= angle_degrees or angle_degrees < 45:
        return "right"
    elif 45 <= angle_degrees < 135:
        return "down"
    elif 135 <= angle_degrees < 225:
        return "left"
    else:
        return "up"

def main():
    for label in LABELS:
        os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

    counters = {label: 0 for label in LABELS}
    cap = cv2.VideoCapture(0)

    print("\n=== Data Collection Guide ===")
    print("Point in one of the four directions.")
    print("Press SPACE to save, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        current_angle = None
        current_direction = None
        landmarks_detected = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]
                index_mcp = hand_landmarks.landmark[5]
                dx = index_tip.x - index_mcp.x
                dy = index_tip.y - index_mcp.y
                angle = math.degrees(math.atan2(dy, dx))
                current_angle = angle
                current_direction = get_direction_from_angle(angle)
                landmarks_detected = True

                tip_x, tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                mcp_x, mcp_y = int(index_mcp.x * frame.shape[1]), int(index_mcp.y * frame.shape[0])
                cv2.arrowedLine(frame, (mcp_x, mcp_y), (tip_x, tip_y), (0, 0, 255), 2)

        # Overlay info
        y_offset = 30
        for label in LABELS:
            count = counters[label]
            color = (0, 255, 0) if count >= SAMPLES_PER_LABEL else (0, 0, 255)
            cv2.putText(frame, f"{label}: {count}/{SAMPLES_PER_LABEL}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30

        if current_direction:
            cv2.putText(frame, f"Current: {current_direction}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status_color = (0, 255, 0) if landmarks_detected else (0, 0, 255)
        status_text = "Landmarks Detected" if landmarks_detected else "No Landmarks"
        cv2.putText(frame, status_text, (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord(' ') and current_direction and landmarks_detected:
            label = current_direction
            if counters[label] < SAMPLES_PER_LABEL:
                filename = f"{label}_{counters[label]}.jpg"
                filepath = os.path.join(DATA_DIR, label, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                counters[label] += 1
                print(f"✓ Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal counts:")
    for label in LABELS:
        print(f"{label}: {counters[label]}/{SAMPLES_PER_LABEL}")

if __name__ == "__main__":
    main()
