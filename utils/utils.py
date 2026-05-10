import os
import cv2
import math
import torch
import numpy as np
from pathlib import Path
from enum import IntFlag, auto
from datetime import datetime

from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class PipelineConfig:
    # --- Required ---
    source_path: str | int
    input_shape: tuple = (640, 640)

    # --- Optional ---
    model_path: str = 'yolov8n.pt'
    frame_count: int = 2
    sleep_second: int = 2
    
    # video_handle: Optional[Any] = None
    target_fps: int = 0

    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4

    # Ultralytics
    use_wrapper: bool = False
    enable_tracking: bool = False

    # Output Options
    is_combine_result: bool = False
    save_video: bool = False
    map_padding: int = 0
    class_id_filter: list = None
    
    # Point picker
    is_manual_picker: bool = True

# Binary Flags
class MediaFilter(IntFlag):
    NONE = 0
    IMAGES = auto()  # 1 (0b001)
    VIDEOS = auto()  # 2 (0b010)
    TEXTS = auto()   # 4 (0b100)
    ALL = IMAGES | VIDEOS | TEXTS 

def rename_media(directory_path, prefix="", mode=MediaFilter.ALL, start_num=1, overwrite=True):
    """
    Renames files with auto-calculated numerical padding. \n
    usage: rename_media("my_folder", prefix="media", mode=(ONLY_IMAGES | ONLY_VIDEOS))
    """
    print(f"--- Running rename_media ---")
    print(f"Looking at: {directory_path}")

    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    video_exts = {'.mp4', '.mkv', '.mov', '.avi', '.wmv'}
    text_exts = {'.txt'}
    
    target_extensions = set()
    if mode & MediaFilter.IMAGES:
        target_extensions.update(image_exts)
        print("Targeting images")
    if mode & MediaFilter.VIDEOS:
        target_extensions.update(video_exts)
        print("Targeting videos")
    if mode & MediaFilter.TEXTS:
        target_extensions.update(text_exts)
        print("Targeting texts")

    path = Path(directory_path)
    if not path.is_dir():
        print(f"Error: {directory_path} is not a valid directory.")
        return

    # Gather and sort files
    files = sorted([f for f in path.iterdir() if f.suffix.lower() in target_extensions])
    total_files = len(files)
    
    if total_files == 0:
        print("No matching files found.")
        return

    # --- Auto-calculate padding ---
    # Calculate based on the highest number that will be reached
    max_number = start_num + total_files - 1
    padding = len(str(max_number)) 
    # Ensure at least 2 digits
    padding = max(padding, 2) 

    if prefix != "":
        prefix += "_"

    print(f"Mode: {bin(mode)} | Found {len(files)} files. Renaming...")
    print("--- PREVIEW ---")
    print(f"Directory: {path.absolute()}")
    print(f"Files found: {total_files}")
    print(f"Example rename: {files[0].name} -> {prefix}{start_num:0{padding}d}{files[0].suffix}")
    print("---------------")

    confirm = input(f"Proceed renaming {total_files} files? (y/n): ").lower()
    if confirm != 'y':
        print("Operation cancelled by user.")
        return

    for i, file_path in enumerate(files, start=start_num):
        extension = file_path.suffix
        
        # Create dynamic format string, e.g., "prefix_0001.jpg"
        new_name = f"{prefix}{i:0{padding}d}{extension}"
        new_path = path / new_name
        
        if new_path.exists() and new_path != file_path and not overwrite:
            print(f"Warning: {new_name} already exists. Skipping to avoid overwrite.")
            continue
            
        file_path.rename(new_path)
        # print(f"Renamed: {file_path.name} -> {new_name}")

def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Expects format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def print_boxed_message(message, width=40):
    inner_width = width - 6
    words = message.split()
    lines = []
    current_line = ""

    # Wrapping logic
    for word in words:
        if not current_line:
            current_line = word
        # Check if adding the next word (plus a space) exceeds the inner_width
        elif len(current_line) + 1 + len(word) <= inner_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    
    # Add the last line
    if current_line:
        lines.append(current_line)

    # Edge case: empty message
    if not lines:
        lines = [""]

    # Print the box
    box_content_width = max(len(line) for line in lines)
    print("=" * (box_content_width + 6))
    for line in lines:
        print(f"|| {line.ljust(box_content_width)} ||")
    print("=" * (box_content_width + 6))

def get_path(path_str: str) -> Path:
    # path_obj = Path(path_str).expanduser().resolve()
    path_obj = Path(path_str)
    return path_obj

def check_path(path_obj: Path):
    if not path_obj.exists():
        raise ValueError("The path {path_obj} does not exist")
    return True

def dict_to_list(data_dict):
    """
    Converts a dict of lists into a list of dicts.
    Uses the shortest list length as the limit.
    """
    # Get the keys and the lists (values)
    keys = data_dict.keys()
    values = data_dict.values()

    # zip(*values) group the elements by 'index' of every list, stop at shortest list length
    return [dict(zip(keys, row)) for row in zip(*values)]

# def save_image(image, folder="output", filename=None, prefix="img"):
#     """
#     Saves an OpenCV image, handles directory creation and naming.
#     """
#     # Validation: avoid empty image
#     if image is None:
#         print("Save Error: Image is empty (None).")
#         return False

#     # Path Handling
#     save_path = Path(folder)
#     save_path.mkdir(parents=True, exist_ok=True)

#     # Naming: If no name is given, use a timestamp
#     if filename is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#         filename = f"{prefix}_{timestamp}.jpg"

#     full_path = save_path / filename
#     success = cv2.imwrite(str(full_path), image)
    
#     if success:
#         # print(f"Successfully saved: {full_path}")
#         None
#     else:
#         # print(f"Failed to save image to: {full_path}")
#         None
        
#     return success

def get_frame_stats(img: np.ndarray):
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

def stitch_two_frames(img1, img2, bg_color=(40, 40, 40)):
    """
    Horizontally stitches two images. If one is shorter, it pads the 
    top and bottom with a background color to perfectly center it.
    bg_color defaults to a sleek dark gray (BGR format).
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    max_h = max(h1, h2)

    # Pad image 1 if it is shorter
    if h1 < max_h:
        diff = max_h - h1
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        img1 = cv2.copyMakeBorder(img1, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=bg_color)

    # Pad image 2 if it is shorter
    if h2 < max_h:
        diff = max_h - h2
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        img2 = cv2.copyMakeBorder(img2, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=bg_color)

    return cv2.hconcat([img1, img2])