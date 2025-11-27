import cv2
import os
import json
import numpy as np

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "trainer.yml")
LABELS_FILE = os.path.join(MODEL_DIR, "labels.json")
CONFIDENCE_THRESHOLD = 80  # lower -> stricter (0 best match); tune as needed

def load_label_map(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}

if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        print("[ERROR] Model file not found. Train model first with train.py")
        exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    label_map = load_label_map(LABELS_FILE)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cam = cv2.VideoCapture(0)
    print("[INFO] Starting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            id_, confidence = recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                name = label_map.get(id_, "Unknown")
                confidence_pct = max(0, min(100, int(100 - confidence)))  # rough percent
                text = f"{name} ({confidence_pct}%)"
            else:
                text = "Unknown"

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
