import cv2
import os
import numpy as np
import json

DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "trainer.yml")
LABELS_FILE = os.path.join(MODEL_DIR, "labels.json")
IMAGE_SIZE = 200  # should match collection size

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_images_and_labels(dataset_dir):
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    persons = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]
    persons.sort()

    for person in persons:
        person_path = os.path.join(dataset_dir, person)
        label_map[current_label] = person
        for fname in os.listdir(person_path):
            fpath = os.path.join(person_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # ensure size
            if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    return faces, labels, label_map

if __name__ == "__main__":
    ensure_dir(MODEL_DIR)
    faces, labels, label_map = load_images_and_labels(DATASET_DIR)
    if len(faces) == 0:
        print("[ERROR] No training images found in dataset/. Capture faces first.")
        exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_FILE)

    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[INFO] Training complete. Model saved to {MODEL_FILE}")
    print(f"[INFO] Labels saved to {LABELS_FILE}")
    print("[INFO] Label map:", label_map)
