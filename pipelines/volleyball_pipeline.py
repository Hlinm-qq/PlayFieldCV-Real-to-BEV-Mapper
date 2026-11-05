import cv2
import numpy as np
from dataclasses import field
from collections import deque

from engine.tracker.court_tracker import *
from engine.detection.yolo_engine import *

from canvas.visualizer import FieldProjector

from data_manager.dataclass import *
from utils.utils import *

class TeamClassifier:
    def __init__(self, court_size: Vector2D = Vector2D(1800, 900), max_history=30):
        self.net_x = court_size.x // 2
        self.court_size = court_size
        self.max_history = max_history
        self.player_colors = {}       # track_id -> deque(maxlen=30)
        self.locked_teams: dict[int, Any] = {}        # track_id -> "Team_A", "Team_B", etc.
        
        # Saved Base Colors (Established in Phase 1)
        self.base_colors = {
            "Team_A": None,
            "Team_B": None
        }
        self.is_bootstrapped = False
        self.margin_dist = 10 # Small margin from court edges
        self.min_player = 2 # Minimum player to classify a team

    def get_jersey_color(self, frame, bbox):
        """Extracts the dominant HSV color from the player's upper torso."""
        x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        
        # Crop the Jersey (Top 20%-50%, Center 60%)
        h, w = y2 - y1, x2 - x1
        crop_y1 = y1 + int(h * 0.2)
        crop_y2 = y1 + int(h * 0.5)
        crop_x1 = x1 + int(w * 0.2)
        crop_x2 = x2 - int(w * 0.2)
        
        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None
            
        jersey_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Convert to HSV and find median color
        hsv_crop = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        median_color = np.median(hsv_crop.reshape(-1, 3), axis=0)
        return median_color

    def assign_teams(self, frame, players: list[PlayerDetection]):
        """Assigns team IDs to players based on jersey color."""
        unclassified_players = []

        for player in players:
            # If already locked, apply cached assignment instantly
            if player.track_id in self.locked_teams:
                player.team_id = self.locked_teams[player.track_id]
                continue

            # Extract color for unclassified player
            color = self.get_jersey_color(frame, player.bbox)
            if color is not None:
                if player.track_id not in self.player_colors:
                    self.player_colors[player.track_id] = deque(maxlen=self.max_history)
                
                self.player_colors[player.track_id].append(color)
                
                # Only attempt to classify once if have at least 5 frames of color history
                if len(self.player_colors[player.track_id]) >= 5:
                    unclassified_players.append(player)

        if not unclassified_players:
            return

        # PHASE 1: BOOTSTRAPPING (Run once at start)
        if not self.is_bootstrapped:
            if len(unclassified_players) >= 4:
                self._bootstrap_base_colors(unclassified_players)
            return

        # PHASE 2: Handles new IDs
        for player in unclassified_players:
            self._classify_single_player(player)

    def _bootstrap_base_colors(self, players: list[PlayerDetection]):
        """Uses spatial layout (the net) to establish the master Team A & B colors."""
        # left_players = [p for p in players if self.margin_dist < p.map_coords.x < self.net_x - self.margin_dist]
        # right_players = [p for p in players if self.net_x + self.margin_dist < p.map_coords.x < self.net_x*2 - self.margin_dist]
        left_players = [p for p in players if p.map_coords.x < self.net_x and self.is_within_court(p.map_coords)]
        right_players = [p for p in players if p.map_coords.x >= self.net_x and self.is_within_court(p.map_coords)]

        if len(left_players) >= self.min_player and len(right_players) >= self.min_player:
            left_colors = [np.mean(self.player_colors[p.track_id], axis=0) for p in left_players]
            right_colors = [np.mean(self.player_colors[p.track_id], axis=0) for p in right_players]

            # Median isolates jersey color from Libero outliers
            self.base_colors["Team_A"] = np.median(left_colors, axis=0)
            self.base_colors["Team_B"] = np.median(right_colors, axis=0)
            self.is_bootstrapped = True

            # Classify these initial bootstrap players
            for p in players:
                self._classify_single_player(p)

    def is_within_court(self, map_coords: Vector2D) -> bool:
        x1 = self.margin_dist
        x2 = self.court_size.x - self.margin_dist
        y1 = self.margin_dist
        y2 = self.court_size.y - self.margin_dist

        if (x1 < map_coords.x < x2) and (y1 < map_coords.y < y2):
            return True

        return False

    def _classify_single_player(self, player: PlayerDetection):
        """Classifies a single player against the known base colors."""
        is_near_net = self.net_x - self.margin_dist < player.map_coords.x < self.net_x + self.margin_dist
        is_out_of_bounds = player.map_coords.y < -50 or player.map_coords.y > self.court_size.y + 50
        
        if is_near_net and is_out_of_bounds:
            self.locked_teams[player.track_id] = "Referee"
            player.team_id = "Referee"
            # print("referee")
            return

        # print("player.map_coords")
        # print(player.track_id, player.map_coords)
        if player.map_coords.y < -200 or player.map_coords.y > self.court_size.y + 200 or \
            player.map_coords.x < -200 or player.map_coords.x > self.court_size.x + 200:
            self.locked_teams[player.track_id] = "Unknown"
            player.team_id = "Unknown"
            # print("unknwodn")
            return

        is_left_side = player.map_coords.x < self.net_x
        player_color = np.mean(self.player_colors[player.track_id], axis=0)

        # Distance to Team A vs Team B base colors
        if is_left_side: # likely team A
            color_delta = np.linalg.norm(player_color - self.base_colors["Team_A"])
            assigned_team = "Team_A" if color_delta < 40 else "Team_A_Libero"
        else: # likely team B
            color_delta = np.linalg.norm(player_color - self.base_colors["Team_B"])
            assigned_team = "Team_B" if color_delta < 40 else "Team_B_Libero"

        # Lock in memory
        self.locked_teams[player.track_id] = assigned_team
        player.team_id = assigned_team
        # print(assigned_team)

    def reset(self):
        """Clean all caches"""
        self.player_colors.clear()
        self.locked_teams.clear()

    def get_locked_team(self, player_id: int):
        return self.locked_teams.get(player_id, "")

class VolleyballPipeline:
    def __init__(self, court_engine: CourtTracker, detector_engine: EngineManager, mapper_handler: FieldProjector, team_classifier: TeamClassifier, run_detect: bool = True):
        self.court = court_engine
        self.detector = detector_engine
        self.mapper = mapper_handler
        self.run_detect = run_detect
        self.team_classifier = team_classifier

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
                team_id="",
                confidence=confidence
            )
            detected_players.append(found_player)

        if self.run_detect and len(detected_players) > 0:
            self.team_classifier.assign_teams(frame, detected_players)

        # Package outputs
        return FrameResult(
            players=detected_players,
            court_corners=detected_corners,
            dst_points=dst_pts,
            homography_matrix=current_H,
            floor_mask=debug_mask
        )