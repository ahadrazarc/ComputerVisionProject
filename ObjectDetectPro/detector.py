# detector.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

class VideoDetector:
    def __init__(self, model_path):
        # Load model once during initialization
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def process(self, video_path, output_path="output_video.mp4"):
        # Open video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        box_color = (0, 255, 255)  # Yellow
        thickness = 2

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, save=False, stream=True)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = box
                    class_name = self.class_names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            out.write(frame)

        cap.release()
        out.release()

        return os.path.abspath(output_path)  # Return the path of saved video
