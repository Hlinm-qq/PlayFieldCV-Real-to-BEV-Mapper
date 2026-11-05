from collections import deque
import numpy as np
import cv2
import math

from canvas.visualizer import FieldProjector
from data_manager.dataclass import *

class DelayBuffer:
    def __init__(self, buffer_size=10, mapper_handler: FieldProjector =None):
        self.buffer_size = buffer_size
        # Stores tuples of (frame, FrameResult)
        self.queue: deque[tuple[np.ndarray, FrameResult]] = deque(maxlen=buffer_size)
        self.mapper = mapper_handler

    def _interpolate_missing_players(self):
        """
        Looks at the newest frame. If a player reappeared after disappearing,
        walks backward through the buffer and backfills their coordinates.
        """
        newest_result = self.queue[-1][1]
        
        for player in newest_result.players:
            track_id = player.track_id
            if track_id == -1: continue # Ignore untracked detections
            team_id = player.team_id

            # Find the last time this ID exists in the older frames
            last_seen_idx = -1
            for i in range(len(self.queue) - 2, -1, -1):
                past_result = self.queue[i][1]
                if any(p.track_id == track_id for p in past_result.players):
                    last_seen_idx = i
                    break
            
            # If they were missing for 1 to (buffer_size - 1) frames, fix it
            frames_missing = (len(self.queue) - 1) - last_seen_idx - 1
            if last_seen_idx != -1 and 0 < frames_missing < self.buffer_size:
                self._backfill(track_id, team_id, last_seen_idx, len(self.queue) - 1)

    def _backfill(self, track_id, team_id, start_idx, end_idx):
        """Calculates the linear step and inserts synthesized players into missing frames."""
        start_player = next(p for p in self.queue[start_idx][1].players if p.track_id == track_id)
        end_player = next(p for p in self.queue[end_idx][1].players if p.track_id == track_id)

        frames_missing = end_idx - start_idx

        # Calculate pixel distance the player traveled while missing
        px_distance = math.hypot(
            end_player.bbox.bottom_center.x - start_player.bbox.bottom_center.x,
            end_player.bbox.bottom_center.y - start_player.bbox.bottom_center.y
        )
        speed_px_per_frame = px_distance / frames_missing

        # Physics sanity check (check if the movement is impossible)
        if speed_px_per_frame > 50.0:
            print(f"Refuse ID interpolation {track_id} - speed too high")
            return

        steps = end_idx - start_idx
        
        # Calculate how much X and Y change per frame
        dx1 = (end_player.bbox.x1 - start_player.bbox.x1) / steps
        dy1 = (end_player.bbox.y1 - start_player.bbox.y1) / steps
        dx2 = (end_player.bbox.x2 - start_player.bbox.x2) / steps
        dy2 = (end_player.bbox.y2 - start_player.bbox.y2) / steps

        # Create the ghost players
        for step, q_idx in enumerate(range(start_idx + 1, end_idx), 1):
            ghost_bbox = BBox(
                x1=start_player.bbox.x1 + (dx1 * step),
                y1=start_player.bbox.y1 + (dy1 * step),
                x2=start_player.bbox.x2 + (dx2 * step),
                y2=start_player.bbox.y2 + (dy2 * step)
            )

            frame_matrix = self.queue[q_idx][1].homography_matrix
            src_coords = ghost_bbox.bottom_center

            # Calculate mapping to 2D canvas
            map_x, map_y = self.mapper.get_map_coordinate(src_coords.x, src_coords.y, frame_matrix)
            
            ghost_player = PlayerDetection(
                track_id=track_id,
                bbox=ghost_bbox,
                src_coords=src_coords,
                team_id=team_id,
                map_coords=Vector2D(map_x, map_y),
                confidence=0.0 # 0.0 marks it as a synthesized frame
            )
            self.queue[q_idx][1].players.append(ghost_player)

    def _smooth_trajectories(self):
        """Applies a Centered Moving Average (CMA) to eliminate jitter."""
        # Only run if the buffer is reasonably full
        if len(self.queue) < 5:
            return

        # Target the frame in the middle of buffer
        mid_idx = len(self.queue) // 2
        target_result = self.queue[mid_idx][1]

        # The smoothing window (e.g., 2 frames back, 2 frames forward)
        window = 2 

        for target_player in target_result.players:
            track_id = target_player.track_id

            if track_id == -1: continue # Ignore untracked target
            # If confidence is 0.0, likely is Ghost player
            if target_player.confidence == 0.0:
                continue

            # Collect the player's coordinates from the surrounding timeline
            x1s, y1s, x2s, y2s = [], [], [], []
            
            for i in range(mid_idx - window, mid_idx + window + 1):
                if 0 <= i < len(self.queue):
                    # Search for the same player in this nearby frame
                    for p in self.queue[i][1].players:
                        if p.track_id == track_id:
                            x1s.append(p.bbox.x1)
                            y1s.append(p.bbox.y1)
                            x2s.append(p.bbox.x2)
                            y2s.append(p.bbox.y2)
                            break
            
            # Apply Centered Average
            if x1s:
                target_player.bbox.x1 = sum(x1s) / len(x1s)
                target_player.bbox.y1 = sum(y1s) / len(y1s)
                target_player.bbox.x2 = sum(x2s) / len(x2s)
                target_player.bbox.y2 = sum(y2s) / len(y2s)
                
                # Re-Map to 2D Canvas
                target_player.src_coords = target_player.bbox.bottom_center
                frame_matrix = target_result.homography_matrix
                if frame_matrix is not None:
                    map_x, map_y = self.mapper.get_map_coordinate(
                        target_player.src_coords.x, 
                        target_player.src_coords.y, 
                        frame_matrix
                    )
                    target_player.map_coords = Vector2D(map_x, map_y)

    def _interpolate_camera_loss(self):
        """Finds gaps where tracking failed and calculates the missing camera movement."""
        end_idx = len(self.queue) - 1
        if self.queue[end_idx][1].court_corners is None:
            return 

        # Look backward to find a gap
        start_idx = -1
        for i in range(end_idx - 1, -1, -1):
            if self.queue[i][1].court_corners is not None:
                start_idx = i
                break

        # If found a gap (e.g., frames missing between start_idx and end_idx)
        frames_missing = end_idx - start_idx - 1
        if start_idx != -1 and frames_missing > 0:
            
            start_dst = self.queue[start_idx][1].dst_points
            end_dst = self.queue[end_idx][1].dst_points

            if not np.array_equal(start_dst, end_dst):
                print("Aborting camera interpolation: dst_points shifted during the gap.")
                return 

            start_corners = self.queue[start_idx][1].court_corners
            end_corners = self.queue[end_idx][1].court_corners
            
            # Calculate the per-frame movement (dx, dy) for all 4 corners simultaneously
            corner_step = (end_corners - start_corners) / (frames_missing + 1)
            
            # Backfill the missing frames with the interpolated corners
            for step, q_idx in enumerate(range(start_idx + 1, end_idx), 1):
                ghost_corners = start_corners + (corner_step * step)
                self.queue[q_idx][1].court_corners = ghost_corners
                self.queue[q_idx][1].dst_points = start_dst

                # print("ghost_corners")
                # print(ghost_corners)
                # print("self.queue[q_idx][1].dst_points")
                # print(self.queue[q_idx][1].dst_points)

                if ghost_corners is None or self.queue[q_idx][1].dst_points is None:
                    print(f"Delay buffer matrix update failed: input array invalid")
                    continue

                # Recalculate the Homography matrix for this frame
                new_H, _ = cv2.findHomography(ghost_corners, self.queue[q_idx][1].dst_points)
                self.queue[q_idx][1].homography_matrix = new_H

    def _smooth_camera(self):
        """Smooths the 4 court corners to eliminate camera jitter."""
        if len(self.queue) < 5: return

        mid_idx = len(self.queue) // 2
        target_result = self.queue[mid_idx][1]
        
        if target_result.court_corners is None:
            return

        window = 2
        # Collect corners from surrounding frames
        corner_history = []
        for i in range(mid_idx - window, mid_idx + window + 1):
            if 0 <= i < len(self.queue) and self.queue[i][1].court_corners is not None:
                corner_history.append(self.queue[i][1].court_corners)
        
        if len(corner_history) > 0:
            # Average the 4 corner points
            smoothed_corners = np.mean(corner_history, axis=0)
            target_result.court_corners = smoothed_corners
            
            # Recalculate a Homography Matrix
            new_H, _ = cv2.findHomography(smoothed_corners, target_result.dst_points)
            target_result.homography_matrix = new_H

    def push(self, frame, result: FrameResult):
        """Adds a new frame to the end of the queue and repairs historical data."""
        self.queue.append((frame, result))
        
        # If have enough history, run repair algorithms across the buffer
        if len(self.queue) > 2:
            self._interpolate_camera_loss()    # If Hough/LK totally failed, interpolate the 4 corners
            self._interpolate_missing_players()

            self._smooth_camera()              # Fix the matrix FIRST
            self._smooth_trajectories()

    def pop(self):
        """Yields the oldest frame (N - buffer_size) which is now fully corrected."""
        if len(self.queue) == self.buffer_size:
            # Pop off the left side of the queue (the oldest frame)
            return self.queue.popleft()
        return None, None

    def drain(self):
        """Yields remaining frames, smoothing them using only past data."""
        while len(self.queue) > 0:
            
            # Target the oldest frame about to be popped
            target_result = self.queue[0][1]
            
            # Smooth it using the remaining frames in the queue
            for target_player in target_result.players:
                if target_player.track_id == -1 or target_player.confidence == 0.0: 
                    continue
                
                x1s, y1s, x2s, y2s = [], [], [], []
                
                # Gather coordinates from whatever is left in the queue
                for _, result in self.queue:
                    for p in result.players:
                        if p.track_id == target_player.track_id:
                            x1s.append(p.bbox.x1)
                            y1s.append(p.bbox.y1)
                            x2s.append(p.bbox.x2)
                            y2s.append(p.bbox.y2)
                            break
                            
                # Apply Trailing Average
                if x1s:
                    target_player.bbox.x1 = sum(x1s) / len(x1s)
                    target_player.bbox.y1 = sum(y1s) / len(y1s)
                    target_player.bbox.x2 = sum(x2s) / len(x2s)
                    target_player.bbox.y2 = sum(y2s) / len(y2s)
                    
                    target_player.src_coords = target_player.bbox.bottom_center
                    frame_matrix = target_result.homography_matrix
                    if frame_matrix is not None:
                        map_x, map_y = self.mapper.get_map_coordinate(
                            target_player.src_coords.x, 
                            target_player.src_coords.y, 
                            frame_matrix
                        )
                        target_player.map_coords = Vector2D(map_x, map_y)

            # Pop and yield tail-end frame
            yield self.queue.popleft()
