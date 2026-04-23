import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
GESTURES = [
    "OPEN_HAND",
    "FIST",
    "POINTING",
    "PEACE",
    "THUMBS_UP",
    "OK"
]

CSV_PATH = "gesture_data.csv"
MODEL_PATH = "gesture_model.pkl"

RECORD_SECONDS = 10

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
# FEATURE ENGINEERING
# -----------------------------
def extract_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

def preprocess(lm):
    base = lm[0]
    return lm - base

def flatten(lm):
    return lm.flatten()

# -----------------------------
# DATA COLLECTION
# -----------------------------
def collect_data():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)    
    
    current_label = 0
    recording = False
    record_start = 0

    print("\nControls:")
    for i, g in enumerate(GESTURES):
        print(f"{i} = {g}")
    print("S = start recording 10 seconds")
    print("0-9 = change label")
    print("ESC = exit\n")

    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["label"] + [f"x{i}" for i in range(42)])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            results = hands.process(rgb)

            label_name = GESTURES[current_label]

            # ALWAYS SHOW WINDOW FIRST
            cv.putText(frame, f"Label: {label_name}", (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv.imshow("Collect Data", frame)

            # KEY INPUT ALWAYS HERE
            key = cv.waitKey(10) & 0xFF

            # EXIT
            if key == 27:
                break

            # CHANGE LABEL
            if ord('0') <= key <= ord('9'):
                num = key - ord('0')
                if num < len(GESTURES):
                    current_label = num
                    print("Switched to:", GESTURES[current_label])

            # SAVE SAMPLE
            if key == ord('s') and not recording:
                recording = True
                record_start = time.time()
                print("Record started: ", label_name)

            if recording:
                elapsed = time.time() - record_start
                remaining = RECORD_SECONDS - elapsed

                cv.putText(frame, "Recording...", (20, 100),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv.putText(frame, f"{remaining:.1f}s left", (20, 140),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                        lm = extract_landmarks(hand_landmarks)
                        lm = preprocess(lm)
                        features = flatten(lm)

                        writer.writerow([label_name, *features])

                if elapsed >= RECORD_SECONDS:
                    recording = False
                    print("Recording finished for: ", label_name) 

            cv.putText(frame, f"Label: {label_name}", (20, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv.imshow("Collect Data", frame)          

    cap.release()
    cv.destroyAllWindows()

# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    if not os.path.exists(CSV_PATH):
        print("No dataset found.")
        return

    X, y = [], []

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "label":
                continue
            y.append(row[0])
            X.append(list(map(float, row[1:])))

    X = np.array(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y_encoded)

    joblib.dump({
        "model": model,
        "label_encoder": le
    }, MODEL_PATH)

    print("Model trained and saved.")

# -----------------------------
# MAIN MENU
# -----------------------------
if __name__ == "__main__":
    print("1 = collect data")
    print("2 = train model")

    choice = input("Select: ")

    if choice == "1":
        collect_data()
    elif choice == "2":
        train_model()