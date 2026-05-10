import os
import cv2
import numpy as np

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
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.jpe')
        self.video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        self.image_paths = []
        self.frame_count = 0
        self.max_frame = 10

        # Check if source is folder of images
        if isinstance(source_path, str) and os.path.isdir(source_path):
            self.image_paths = sorted([
                os.path.join(source_path, f) for f in os.listdir(source_path)
                if f.lower().endswith(self.image_extensions)
            ])
            if not self.image_paths:
                raise ValueError(f"No images found in directory: {source_path}")

            self.source_type = "folder"
        # Check if source is a single image
        elif isinstance(source_path, str) and source_path.lower().endswith(self.image_extensions):
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Image not found: {source_path}")
            self.image_paths = [source_path]
            self.source_type = "image"
        # Treat as Video file
        else:
            self.capture = cv2.VideoCapture(source_path)
            if not self.capture.isOpened():
                raise ValueError(f"Failed to open video source: {source_path}")
            self.source_type = "video"

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

    def get_fps(self):
        # Video file. 
        # Read its native metadata
        if self.source_type == "video":
            return self.capture.get(cv2.CAP_PROP_FPS)

    def get_first_frame(self):
        """
        Get the first frame of given source
        """
        # Is image(s) 
        if self.source_type in ["folder", "image"]:
            for path in self.image_paths:
                if self.frame_count > self.max_frame:
                    break
                
                frame = cv2.imread(path)
                if frame is not None:
                    self.frame_count += 1
                    return frame
            return
        # Is Video stream
        elif self.source_type == "video":
            while True and self.frame_count < self.max_frame:
                success, frame = self.capture.read()
                if not success:
                    break
                self.frame_count += 1
                return frame
            
    def get_frame_stats(self, img: np.ndarray):
        h, w, c = img.shape
        stats = {
            "Dimensions": (w, h),
            "Total Pixels": img.size // c,
            "Data Type": img.dtype,
            "Min": np.min(img),
            "Max": np.max(img),
            "Mean": np.mean(img),
            "Std Dev": np.std(img)
        }
        return stats