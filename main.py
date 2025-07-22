from flask import Flask, jsonify
import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__)

# ========== EYE DETECTION ==========
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# ========== ENCODE KNOWN FACES ==========
def encode_faces(dataset_path='dataset'):
    known_faces = []
    known_names = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        known_faces.append(encodings[0])
                        label = os.path.basename(os.path.dirname(img_path))
                        known_names.append(label)
                        print(f"Encoded: {img_path} as {label}")
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
    return known_faces, known_names

print("Encoding faces from dataset...")
known_faces, known_names = encode_faces()
if not known_faces:
    print("Warning: No known faces found. Exiting.")
    exit()

# ========== IN-MEMORY ATTENDANCE RECORD ==========
attendance_records = []

# ========== FLASK ROUTES ==========

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'message': '✅ Server is running'}), 200

@app.route('/start-camera', methods=['GET'])
def start_camera():
    cap = cv2.VideoCapture(0)
    FACE_DISTANCE_THRESHOLD = 0.35
    print("Starting camera...")

    if not cap.isOpened():
        return jsonify({'match': False, 'message': 'Camera not accessible'}), 500

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = [v * 4 for v in face_location]
            face_img = frame[top:bottom, left:right]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Eye detection
            eyes_detected = False
            landmarks = face_recognition.face_landmarks(rgb_frame, [face_location])
            if landmarks:
                left_eye = landmarks[0].get("left_eye", [])
                right_eye = landmarks[0].get("right_eye", [])
                if len(left_eye) >= 4 and len(right_eye) >= 4:
                    eyes_detected = True
            if not eyes_detected:
                eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
                if len(eyes) >= 2:
                    eyes_detected = True

            if not eyes_detected:
                print("Eyes not detected, skipping...")
                continue

            now = datetime.now()
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')

            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                confidence = (1 - best_distance) * 100

                if best_distance < FACE_DISTANCE_THRESHOLD:
                    name = known_names[best_match_index]

                    image_dir = 'attendance_faces'
                    os.makedirs(image_dir, exist_ok=True)
                    filename = f"{image_dir}/{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, face_img)

                    attendance_records.append({
                        'name': name,
                        'image_path': filename,
                        'timestamp': now_str
                    })

                    cap.release()
                    cv2.destroyAllWindows()

                    return jsonify({
                        'match': True,
                        'name': name,
                        'timestamp': now_str,
                        'confidence': round(confidence, 2),
                        'image_saved': filename,
                        'message': f'{name} matched with confidence {confidence:.2f}% ✅'
                    })

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'match': False, 'message': 'No matching face found'}), 200

# ========== START FLASK APP ==========
if __name__ == '__main__':
    app.run(debug=True)
