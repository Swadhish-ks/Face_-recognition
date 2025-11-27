---
# **Face Recognition Using OpenCV**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

[![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0.72-brightgreen)](https://opencv.org/)

[![License](https://img.shields.io/badge/License-Open%20for%20Personal%20Use-yellow)](#license)

---

## **Project Overview**

This project implements a **real-time face recognition system** using **Python** and **OpenCV**. The system captures face images from your webcam, trains a **LBPH (Local Binary Patterns Histogram)** face recognizer, and identifies faces in real-time.

**Key Features:**

* Real-time face detection with webcam
* LBPH-based face recognition
* Multi-user support
* Saves trained model (`trainer.yml`) and label mapping (`labels.json`)
* Dataset preprocessing with **200×200 padded images** for consistent recognition

---

## **Project Structure**

```
face_recognition_project/
├── dataset/                # Folders for each person with images
├── models/                 # Trained model and label map
│   ├── trainer.yml
│   └── labels.json
├── collect_faces.py        # Capture face images
├── train.py                # Train LBPH model
├── recognize.py            # Real-time recognition
└── README.md               # Project documentation
```

---

## **Installation**

1. Clone the repository:

```bash
git clone <your-repo-url>
cd face_recognition_project
```

2. Create a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install opencv-contrib-python==4.7.0.72 numpy
```

> ⚠ Use Python 3.7–3.11 for compatibility with OpenCV-Contrib.

---

## **Usage**

### **Step 1 — Collect Face Images**

```bash
python collect_faces.py --name alice --samples 50
```

* `--name` → folder name for the person
* `--samples` → number of images to capture
* Press **q** to stop early

> Repeat for each person you want to add.

---

### **Step 2 — Train the Model**

```bash
python train.py
```

This creates:

* `models/trainer.yml` → LBPH face recognition model
* `models/labels.json` → ID-to-name mapping

---

### **Step 3 — Real-Time Recognition**

```bash
python recognize.py
```

* Opens webcam, detects faces, displays name & confidence
* Press **q** to quit

---

## **Files Explained**

| File                 | Purpose                            |
| -------------------- | ---------------------------------- |
| `collect_faces.py`   | Capture images for a person        |
| `train.py`           | Train LBPH model                   |
| `recognize.py`       | Real-time recognition using webcam |
| `models/trainer.yml` | Stores trained LBPH features       |
| `models/labels.json` | Maps numeric labels to names       |
| `dataset/`           | Contains folders of person images  |

---

## **Tips for Best Accuracy**

* Collect **30–50 images per person**
* Ensure **good lighting** and frontal face images
* To add a new user: add images → rerun `train.py`
* Adjust `CONFIDENCE_THRESHOLD` in `recognize.py` for stricter recognition

---

## **Future Improvements**

* DNN-based face detection for higher accuracy
* Deep-learning embeddings (FaceNet / ArcFace)
* GUI for user management and dataset collection
* Log recognized faces for attendance system

---

## **License**

Open for personal and educational use.

---

If you want, I can **also create a version with GitHub badges + a screenshot of the recognition window** so it looks fully polished for your repository homepage.

Do you want me to do that next?
# Face_-recognition
