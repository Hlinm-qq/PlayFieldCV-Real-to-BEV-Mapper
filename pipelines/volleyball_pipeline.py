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
        raw_detections = []

        if self.run_detect:
            results = self.detector.detect(frame)
            raw_detections = dict_to_list(results)

        # Update Court Matrix
        current_H, debug_mask, detected_corners, dst_pts = self.court.update(frame, frame_idx, raw_detections)

        # print("dst_pts")
        # print(dst_pts)
        # print("detected_corners")
        # print(detected_corners)

        # if (self.run_detect):
        #     # Run Object Detection
        #     results = self.detector.detect(frame)
        #     detections = dict_to_list(results)

        # Map Player Coordinates
        for det in raw_detections:
            x1, y1, x2, y2 = det["xyxy"]
            new_bbox = BBox(x1, y1, x2, y2)

            confidence = det["confidence"]
            tracked_id = det["id"]

            # Calculate foot coordinate (bottom center of bounding box)
            src_coords = new_bbox.bottom_center

            # map_x, map_y = self.mapper.get_map_coordinate(foot_x, foot_y, current_H)
            map_x, map_y = self.mapper.get_map_coordinate(src_coords.x, src_coords.y, current_H)
            map_coords = Vector2D(map_x, map_y)

            found_player = PlayerDetection(
                track_id=tracked_id, 
                bbox=new_bbox, 
                src_coords=src_coords, 
                map_coords=map_coords, 
                confidence=confidence
            )
            detected_players.append(found_player)

        # Package outputs
        return FrameResult(
            players=detected_players,
            court_corners=detected_corners,
            dst_points=dst_pts,
            homography_matrix=current_H,
            floor_mask=debug_mask
        )