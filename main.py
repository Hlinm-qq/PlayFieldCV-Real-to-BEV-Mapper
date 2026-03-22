import os
import math
import torch
import cv2
import glob
import random
import torchvision
import time

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from enum import Enum

from ultralytics import YOLO

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

print(torch.cuda.is_available())  # True if GPU detected

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join(".", "dataset")
archive_dir = os.path.join("..", "dataset")
pascal_2012_dir = os.path.join(archive_dir, "VOC2012", "JPEGImages")

class UnifiedSource:
    def __init__(self, source_path):
        """
        source_path can be:
        - 0 (integer): local webcam
        - *.mp4 (String): Video file
        - *.jpg (String): Single image file
        - "dataset/frames/" (String): Folder of images
        """
        self.source_type = None
        self.capture = None
        self.valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = []
        self.frame_count = 0
        self.max_frame = 10

        # Check if source is folder of images
        if isinstance(source_path, str) and os.path.isdir(source_path):
            self.source_type = "folder"
            self.image_paths = sorted([
                os.path.join(source_path, f) for f in os.listdir(source_path)
                if f.lower().endswith(self.valid_image_extensions)
            ])

            if not self.image_paths:
                raise ValueError(f"No images found in directory: {source_path}")
        # Check if source is a single image
        elif isinstance(source_path, str) and source_path.lower().endswith(self.valid_image_extensions):
            self.source_type = "image"
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Image not found: {source_path}")
            self.image_paths = [source_path]
        # Treat as Video file
        else:
            self.source_type = "video"
            self.capture = cv2.VideoCapture(source_path)
            if not self.capture.isOpened():
                raise ValueError(f"Failed to open video source: {source_path}")

    def get_frame(self):
        """
        Generator that streams the data
        """
        # Is image(s) 
        if self.source_type in ["folder", "image"]:
            for path in self.image_paths:
                if self.frame_count > self.max_frame:
                    break
                
                frame = cv2.imread(path)
                if frame is not None:
                    self.frame_count += 1
                    yield frame
            return
        # Is Video stream
        elif self.source_type == "video":
            while True and self.frame_count < self.max_frame:
                success, frame = self.capture.read()
                if not success:
                    break
                self.frame_count += 1
                yield frame
    
    def get_type(self):
        return self.source_type
    
    def set_max_frame(self, val: int):
        self.max_frame = val
        
    def close(self):
        if self.capture is not None:
            self.capture.release()

class EngineType(Enum):
    ULTRALYTICSYOLO = 1
    OTHER = 2

def load_model(type: EngineType):
    if type == EngineType.ULTRALYTICSYOLO:
        return YOLO('yolov8n.pt')

source_path = pascal_2012_dir
frame_count = 5

# Initialize unified stream reader
stream = UnifiedSource(source_path)
stream.set_max_frame(frame_count)
is_live_stream = isinstance(source_path, int) and source_path == 0
is_images = True if stream.get_type() != "video" else False

# Determine what FPS we are trying to follow
target_fps = 0
if is_live_stream:
    target_fps = 0 # Live streams run AFAP on the newest frame, no throttling
elif is_images:
    # Set for visual playback
    # target_fps = min(30, frame_count) 
    target_fps = 0
else:
    # Video file. Read its native metadata.
    temp_cap = cv2.VideoCapture(source_path)
    target_fps = temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()

# Calculate the sleep target
target_frame_time = (1.0 / target_fps) if target_fps > 0 else 0

model = load_model(EngineType.ULTRALYTICSYOLO)
model.to(device)
print(model.device)

# --- PIPELINE LOOP ---
for frame in stream.get_frame():
    # frame = cv2.imread("dummy.jpg") # Placeholder for stream.get_frame()
    
    loop_start_time = time.time()

    # ==========================================
    # A. EXECUTE THE AI (Preprocess -> Engine -> Postprocess)
    # ==========================================
    if not is_images and not is_live_stream:
        results = model(frame, stream=True)
    else:
        results = model(frame)

    for result in results:
        # result extraction
        boxes = result.boxes

        if boxes is not None:
            xywh_array = boxes.xywh.cpu().numpy()
            confidences = boxes.conf.cpu().numpy() # confidence score for each box
            names = [result.names[id.item()] for id in boxes.cls.int().cpu().numpy()] # class name of each box
            # names = [id.item() for id in boxes.cls.int().cpu().numpy()] # class name of each box
            # print(names)

            for (cx, cy, w, h), conf, name in zip(xywh_array, confidences, names):
                print(f"Frame object -> Class: {name} | Conf: {conf:.2f} | X:{cx:.1f} Y:{cy:.1f} W:{w:.1f} H:{h:.1f}")

                x_top_left = int(cx - (w / 2))
                y_top_left = int(cy - (h / 2))

                cv2.rectangle(frame, (x_top_left, y_top_left), (x_top_left + int(w), y_top_left + int(h)), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (x_top_left, y_top_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # ==========================================
    # B. MEASURE PIPELINE SPEED
    # ==========================================
    process_time = time.time() - loop_start_time

    # ==========================================
    # C. THROTTLE (If necessary)
    # ==========================================
    # If we are processing a video/folder AND our AI is faster than the target FPS
    if target_frame_time > 0 and process_time < target_frame_time:
        sleep_amount = target_frame_time - process_time
        time.sleep(sleep_amount)

    # ==========================================
    # D. CALCULATE FINAL END-TO-END FPS
    # ==========================================
    total_loop_time = time.time() - loop_start_time
    actual_fps = 1.0 / total_loop_time if total_loop_time > 0 else 0

    # Draw telemetry on screen
    cv2.putText(frame, f"Target FPS: {target_fps if target_fps > 0 else 'MAX'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Actual FPS: {int(actual_fps)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Window", 1280, 720)
    cv2.imshow("Window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.waitKey(1000 * 10)
    # time.sleep(15)

stream.close()
cv2.destroyAllWindows()






