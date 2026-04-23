# Hand Gesture Recognition (MediaPipe + ML)

A real-time hand gesture recognition system using **MediaPipe Hands** and a **machine learning classifier (RandomForest)**.  
Supports basic hand gestures like open hand, fist, peace, thumbs up, pointing, and OK.

---

## Features

- Real-time hand tracking using MediaPipe
- Custom dataset collection tool
- Train your own gesture classifier
- Supports 6 basic gestures:
  - OPEN_HAND
  - FIST
  - POINTING
  - PEACE
  - THUMBS_UP
  - OK
- Lightweight ML model (scikit-learn RandomForest)
- Works on webcam input

---

## 📦 Requirements

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scikit-learn joblib
