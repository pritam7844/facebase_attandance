import face_recognition
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

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

print(" Encoding faces from dataset...")
known_faces, known_names = encode_faces()

if not known_faces:
    print(" Warning: No known faces found. Exiting.")
    exit()

attendance = {}
FACE_DISTANCE_THRESHOLD = 0.5

cap = cv2.VideoCapture(0)
print(" Starting camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        name = "Unknown"
        top, right, bottom, left = [v * 4 for v in face_location]

        face_img = frame[top:bottom, left:right]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        face_landmarks_list = face_recognition.face_landmarks(rgb_frame, [face_location])
        eyes_detected = False
        if face_landmarks_list:
            landmarks = face_landmarks_list[0]
            left_eye = landmarks.get('left_eye', [])
            right_eye = landmarks.get('right_eye', [])
            if len(left_eye) >= 4 and len(right_eye) >= 4:
                eyes_detected = True

        if not eyes_detected:
            eyes = eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(30, 30)
            )
            if len(eyes) >= 2:
                eyes_detected = True

        if not eyes_detected:
            now = datetime.now()
            timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
            unknown_dir = 'unknown_faces'
            os.makedirs(unknown_dir, exist_ok=True)
            filename = f"{unknown_dir}/unknown_{timestamp}.jpg"
            cv2.imwrite(filename, face_img)
            print(f"Face ignored due to insufficient eyes detected, saved as {filename}")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "No Eyes Detected", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            continue 

        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            print(f"Best match distance: {best_distance:.3f} for {known_names[best_match_index]}")

            if best_distance < FACE_DISTANCE_THRESHOLD:
                name = known_names[best_match_index]
                if name not in attendance:
                    time_str = datetime.now().strftime('%H:%M:%S')
                    attendance[name] = time_str
                    print(f"{name} marked present at {time_str}")
            else:
                now = datetime.now()
                timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
                unknown_dir = 'unknown_faces'
                os.makedirs(unknown_dir, exist_ok=True)
                filename = f"{unknown_dir}/unknown_{timestamp}.jpg"
                cv2.imwrite(filename, face_img)
                print(f" Unknown face saved as: {filename}")
        else:
            now = datetime.now()
            timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
            unknown_dir = 'unknown_faces'
            os.makedirs(unknown_dir, exist_ok=True)
            filename = f"{unknown_dir}/unknown_{timestamp}.jpg"
            cv2.imwrite(filename, face_img)
            print(f"Unknown face saved as: {filename}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
df = pd.DataFrame(list(attendance.items()), columns=["Name", "Time"])
df.to_csv('attendance.csv', index=False)
print("\n Attendance saved to 'attendance.csv'")