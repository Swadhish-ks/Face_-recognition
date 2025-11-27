import cv2
import os
import argparse

def resize_with_padding(gray_image, size=200):
    h, w = gray_image.shape[:2]
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(gray_image, (new_w, new_h))

    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def collect(name, samples=50, camera_index=0, size=200):
    dataset_dir = "dataset"
    person_dir = os.path.join(dataset_dir, name)
    ensure_dir(person_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(camera_index)
    count = 0
    print(f"[INFO] Starting capture for '{name}'. Press 'q' to quit early.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = resize_with_padding(face_img, size=size)
            count += 1
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Collected: {count}/{samples}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Collect Faces", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= samples:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finished. Collected {count} images for '{name}' in {person_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect face images for a person")
    parser.add_argument("--name", required=True, help="Person name (used as folder name)")
    parser.add_argument("--samples", type=int, default=50, help="Number of face samples to collect")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--size", type=int, default=200, help="Output image size (square), default 200")
    args = parser.parse_args()

    collect(args.name, samples=args.samples, camera_index=args.camera, size=args.size)
