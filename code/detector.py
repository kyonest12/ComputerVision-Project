import cv2
from ultralytics import YOLO
import numpy as np

class HOGPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))
        detections = []

        for (x, y, w, h), weight in zip(boxes, weights):
            if weight > 0.6:
                detections.append(([x, y, w, h], float(weight), 'person'))

        return detections

class YOLOPersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path) 

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]
            if label == "person" and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

        return detections