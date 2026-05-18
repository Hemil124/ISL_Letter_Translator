import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model & labels
model = load_model("isl_letter_model_onehand.h5")
labels = np.load("classesOneHand.npy")

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,  # single hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    predicted_letter = ""

    # SINGLE HAND extraction
    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract 63 features
        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.extend([lm.x, lm.y, lm.z])

        if len(landmark_list) == 63:
            data = np.array(landmark_list).reshape(1, -1)

            prediction = model.predict(data, verbose=0)
            class_id = np.argmax(prediction)

            # predicted_letter = labels[class_id]

            confidence = np.max(prediction)
            predicted_letter = f"{labels[class_id]} ({confidence:.2f})"

    cv2.putText(
        frame,
        f"Letter: {predicted_letter}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Single-Hand ISL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
