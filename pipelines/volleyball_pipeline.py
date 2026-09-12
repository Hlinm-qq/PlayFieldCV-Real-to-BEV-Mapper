import cv2
import numpy as np
from dataclasses import field

from engine.tracker.court_tracker import *
from engine.detection.yolo_engine import *

from canvas.visualizer import FieldProjector

from data_manager.dataclass import *
from utils.utils import *

class VolleyballPipeline:
    def __init__(self, court_engine: CourtTracker, detector_engine: EngineManager, mapper_handler: FieldProjector, run_detect: bool = True):
        self.court = court_engine
        self.detector = detector_engine
        self.mapper = mapper_handler
        self.run_detect = run_detect

    def initialize(self, frame, src_corners, map_dst_points):
        self.court.initialize(frame, src_corners, map_dst_points)

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        # metrics = FrameMetrics()
        detected_players = []

        # Update Court Matrix
        current_H, debug_mask = self.court.update(frame, frame_idx)

        if (self.run_detect):
            # Run Object Detection
            results = self.detector.detect(frame)
            detections = dict_to_list(results)

            # Map Player Coordinates
            for det in detections:
                x1, y1, x2, y2 = det["xyxy"]
                new_bbox = BBox(x1, y1, x2, y2)

                # Calculate foot coordinate (bottom center of bounding box)
                src_coords = new_bbox.bottom_center

                # map_x, map_y = self.mapper.get_map_coordinate(foot_x, foot_y, current_H)
                map_x, map_y = self.mapper.get_map_coordinate(src_coords.x, src_coords.y, current_H)
                map_coords = Vector2D(map_x, map_y)

                found_player = PlayerDetection(
                    track_id=-1, 
                    bbox=new_bbox, 
                    src_px=src_coords, 
                    map_px=map_coords, 
                    confidence=0.0
                )
                detected_players.append(found_player)

        # Package outputs
        return FrameResult(
            players=detected_players,
            homography_matrix=current_H,
            floor_mask=debug_mask
        )