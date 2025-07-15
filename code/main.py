import cv2
import numpy as np
import imutils
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Khởi tạo bộ phát hiện người HOG ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

MIN_PERSON_WIDTH = 60
MIN_PERSON_HEIGHT = 120
BOUNDING_BOX_COLOR = (0, 255, 0)
PATH_COLOR = (0, 0, 255)
PATH_THICKNESS = 2

# --- Khởi tạo DeepSORT tracker ---
tracker = DeepSort(max_age=30, n_init=3)

# --- Đọc video ---
video_path = 'video nguoi + cho.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Lỗi: Không thể mở video '{video_path}'")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = imutils.resize(frame, width=600)

    # --- Phát hiện người bằng HOG ---
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(32, 32), scale=1.05)

    detections = []
    for i, (x, y, w, h) in enumerate(rects):
        if w > MIN_PERSON_WIDTH and h > MIN_PERSON_HEIGHT:
            detections.append(([x, y, w, h], 1.0, 'person'))

    # --- Cập nhật theo dõi ---
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # --- Vẽ kết quả ---
    for track in tracks:
        print("cccccc")
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # (left, top, right, bottom)
        x1, y1, x2, y2 = map(int, ltrb)
        print(x1, y1, x2, y2)
        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOUNDING_BOX_COLOR, 2)

        # Vẽ ID
        cv2.putText(frame, f"ID: {track_id}", (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PATH_COLOR, 2)


    # --- Hiển thị khung hình ---
    cv2.imshow("HOG + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Dọn dẹp ---
cap.release()
cv2.destroyAllWindows()
print(f"Đã xử lý {frame_count} khung hình.")
