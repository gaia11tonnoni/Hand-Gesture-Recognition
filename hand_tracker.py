import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "gesture_model.pkl"

data = joblib.load(MODEL_PATH)
model = data["model"]
label_encoder = data["label_encoder"]

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# SMOOTHING BUFFER
# -----------------------------
prediction_history = deque(maxlen=10)

# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------
def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

def preprocess(landmarks):
    base_x, base_y = landmarks[0]
    return landmarks - [base_x, base_y]

def flatten(landmarks):
    return landmarks.flatten().reshape(1, -1)

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = hands.process(rgb)

        gesture = "NO HAND"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # feature pipeline
                landmarks = extract_landmarks(hand_landmarks)
                landmarks = preprocess(landmarks)
                features = flatten(landmarks)

                # prediction (FIXED PIPELINE)
                pred_encoded = model.predict(features)[0]
                pred_label = label_encoder.inverse_transform([pred_encoded])[0]

                # smoothing
                prediction_history.append(pred_label)
                gesture = Counter(prediction_history).most_common(1)[0][0]

        cv.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv.imshow("Hand Tracker", frame)

        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()