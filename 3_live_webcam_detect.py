"""
STEP 3: Live Webcam Emotion Detection (OpenCV + Trained CNN)
--------------------------------------------------------------
Ye script laptop ke webcam se live video lega, face detect karega,
aur trained model se real-time emotion predict karke screen par
bounding box + label dikhayega.

Run: python 3_live_webcam_detect.py
Quit: 'q' key press karke
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Trained Model Load Karein
print("Model load ho raha hai, kripya wait karein...")
model = load_model("optimal_emotion_model.h5")
print("Model successfully load ho gaya!")

# 2. OpenCV Face Detector (Haar Cascade)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 3. Emotion Classes (FER-2013 alphabetical order — must match class_indices from training)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# 4. Webcam Start (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam access nahi ho raha hai!")
    exit()

print("Webcam start ho gaya hai. Band karne ke liye 'q' press karein.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture nahi ho pa raha hai.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            confidence = f"{round(np.max(prediction) * 100, 1)}%"
            display_text = f"{label} ({confidence})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, display_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

    cv2.imshow("Facial Emotion Recognition - Live Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()