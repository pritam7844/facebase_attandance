import face_recognition
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

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

# ========== START CAMERA ==========
cap = cv2.VideoCapture(0)
FACE_DISTANCE_THRESHOLD = 0.35
print("Starting camera. Press 'q' to quit.")

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
        today_str = now.strftime('%Y-%m-%d')

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]
            confidence = (1 - best_distance) * 100

            print(f"[MATCH DEBUG] Best: {known_names[best_match_index]} - Distance: {best_distance:.4f} - Confidence: {confidence:.2f}%")

            if best_distance < FACE_DISTANCE_THRESHOLD:
                name = known_names[best_match_index]

                # Filter today's records for this name
                todays_records = [r for r in attendance_records if r['name'] == name and r['timestamp'].startswith(today_str)]

                if len(todays_records) >= 2:
                    print(f"{name} already has 2 attendance entries today. Skipping.")
                    continue

                elif len(todays_records) == 1:
                    last_time = datetime.strptime(todays_records[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if (now - last_time) < timedelta(hours=2):
                        print(f"{name} marked recently (<2 hrs). Skipping.")
                        continue

                image_dir = 'attendance_faces'
                os.makedirs(image_dir, exist_ok=True)
                filename = f"{image_dir}/{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, face_img)

                attendance_records.append({
                    'name': name,
                    'image_path': filename,
                    'timestamp': now_str
                })

                print(f"{name} ✅ Attendance recorded at {now_str}")

                # Draw on frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label_text = f"{name} ({confidence:.1f}%)"
                cv2.putText(frame, label_text, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                print("Face not recognized (distance too high), skipping...")
        else:
            print("No faces in dataset to compare with.")

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ========== EXPORT TO CSV ==========
df = pd.DataFrame(attendance_records)
if not df.empty:
    df['ID'] = range(1, len(df) + 1)
    df[['ID']].to_csv('attendance_database_export.csv', index=False)
    print("\n✅ Attendance exported to 'attendance_database_export.csv' (only IDs)")
else:
    print("\n⚠️ No attendance recorded.")
