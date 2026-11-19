from collections import deque
import numpy as np
import cv2
import math

from canvas.visualizer import FieldProjector
from data_manager.dataclass import *

class DelayBuffer:
    def __init__(self, buffer_size=10, mapper_handler: FieldProjector =None, court_size=(1800, 900)):
        self.buffer_size = buffer_size
        # Stores tuples of (frame, FrameResult)
        self.queue: deque[tuple[np.ndarray, FrameResult]] = deque(maxlen=buffer_size)
        self.mapper = mapper_handler

        # Static destination reference points for the entire match
        self.base_dst_points = np.array([
            [0, 0],
            [court_size[0], 0],
            [court_size[0], court_size[1]],
            [0, court_size[1]]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # 5-frame normalized smoothing kernel
        self.gaussian_weights = np.array([0.061, 0.242, 0.383, 0.242, 0.061], dtype=np.float32)

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

    def _interpolate_camera_gaps(self):
        """Fills tracking dropped frames linearly between valid camera keyframes."""
        end_idx = len(self.queue) - 1
        if self.queue[end_idx][1].court_corners is None:
            return

        # Find previous valid frame
        start_idx = -1
        for i in range(end_idx - 1, -1, -1):
            if self.queue[i][1].court_corners is not None:
                start_idx = i
                break

        gap = end_idx - start_idx - 1
        if start_idx != -1 and gap > 0:
            c_start = self.queue[start_idx][1].court_corners
            c_end = self.queue[end_idx][1].court_corners
            step = (c_end - c_start) / (gap + 1)

            for s, q_idx in enumerate(range(start_idx + 1, end_idx), 1):
                interp_corners = c_start + (step * s)
                self.queue[q_idx][1].court_corners = interp_corners
                H, _ = cv2.findHomography(interp_corners, self.base_dst_points)
                self.queue[q_idx][1].homography_matrix = H

    def _smooth_camera_trajectory(self):
        """Applies a 5-tap Gaussian window to the canonical corners of the middle frame."""
        mid_idx = len(self.queue) // 2
        target_res = self.queue[mid_idx][1]

        # Ensure full 5-frame local window exists around mid_idx
        window_indices = list(range(mid_idx - 2, mid_idx + 3))
        if any(idx < 0 or idx >= len(self.queue) for idx in window_indices):
            return

        sampled_corners = []
        for idx in window_indices:
            corners = self.queue[idx][1].court_corners
            if corners is None:
                return  # Skip smoothing if window contains unresolved drops
            sampled_corners.append(corners)

        # Weighted temporal average: sum(w_i * corner_i)
        sampled_corners = np.array(sampled_corners)  # shape (5, 4, 1, 2)
        smoothed_corners = np.sum(sampled_corners * self.gaussian_weights[:, None, None, None], axis=0)

        target_res.court_corners = smoothed_corners.astype(np.float32)
        H_smooth, _ = cv2.findHomography(target_res.court_corners, self.base_dst_points)
        target_res.homography_matrix = H_smooth

    def _interpolate_missing_players(self):
        """Interpolates short occlusions (1-3 frames) in image space."""
        newest_result = self.queue[-1][1]
        for p in newest_result.players:
            tid = p.track_id
            if tid == -1:
                continue

            last_seen = -1
            for i in range(len(self.queue) - 2, -1, -1):
                if any(x.track_id == tid for x in self.queue[i][1].players):
                    last_seen = i
                    break

            missing = (len(self.queue) - 1) - last_seen - 1
            if last_seen != -1 and 0 < missing <= 3:
                p_start = next(x for x in self.queue[last_seen][1].players if x.track_id == tid)
                p_end = p

                # Guard against false cross-court ID swaps
                dist = math.hypot(
                    p_end.src_coords.x - p_start.src_coords.x,
                    p_end.src_coords.y - p_start.src_coords.y
                )
                if (dist / (missing + 1)) > 50.0:
                    continue

                dx1 = (p_end.bbox.x1 - p_start.bbox.x1) / (missing + 1)
                dy1 = (p_end.bbox.y1 - p_start.bbox.y1) / (missing + 1)
                dx2 = (p_end.bbox.x2 - p_start.bbox.x2) / (missing + 1)
                dy2 = (p_end.bbox.y2 - p_start.bbox.y2) / (missing + 1)

                for step, q_idx in enumerate(range(last_seen + 1, len(self.queue) - 1), 1):
                    interp_bbox = BBox(
                        p_start.bbox.x1 + dx1 * step,
                        p_start.bbox.y1 + dy1 * step,
                        p_start.bbox.x2 + dx2 * step,
                        p_start.bbox.y2 + dy2 * step
                    )
                    src_coords = interp_bbox.bottom_center
                    ghost = PlayerDetection(
                        track_id=tid,
                        bbox=interp_bbox,
                        src_coords=src_coords,
                        team_id=p.team_id,
                        map_coords=Vector2D(0, 0),  # Calculated on pop()
                        confidence=0.0
                    )
                    self.queue[q_idx][1].players.append(ghost)

    def _smooth_player_trajectories(self):
        """Smooths raw image bounding boxes with a centered 3-tap window."""
        mid_idx = len(self.queue) // 2
        target_players = self.queue[mid_idx][1].players

        for tp in target_players:
            tid = tp.track_id
            if tid == -1 or tp.confidence == 0.0:
                continue

            history_boxes = []
            for i in [mid_idx - 1, mid_idx, mid_idx + 1]:
                if 0 <= i < len(self.queue):
                    match = next((p for p in self.queue[i][1].players if p.track_id == tid), None)
                    if match:
                        history_boxes.append([match.bbox.x1, match.bbox.y1, match.bbox.x2, match.bbox.y2])

            if len(history_boxes) == 3:
                avg = np.mean(history_boxes, axis=0)
                tp.bbox.x1, tp.bbox.y1, tp.bbox.x2, tp.bbox.y2 = avg
                tp.src_coords = tp.bbox.bottom_center

    def _project_frame_to_bev(self, result):
        """Projects ALL players using the finalized matrix."""
        if result.homography_matrix is None or self.mapper is None:
            return

        for player in result.players:
            map_x, map_y = self.mapper.get_map_coordinate(
                player.src_coords.x,
                player.src_coords.y,
                result.homography_matrix
            )
            player.map_coords = Vector2D(map_x, map_y)

    def _smooth_tail_frames(self):
        """Applies bounded centered smoothing to tail frames that never reached mid_idx."""
        n = len(self.queue)
        if n < 2:
            return

        # Frames up to mid_idx were already smoothed during push()
        start_tail = (n // 2) + 1 if n >= 5 else 0

        for idx in range(start_tail, n):
            target_res = self.queue[idx][1]
            # Bounded 3-tap window (at most 1 frame back, 1 frame forward)
            win_indices = [i for i in (idx - 1, idx, idx + 1) if 0 <= i < n]

            # Smooth tail camera corners & recompute H
            if target_res.court_corners is not None:
                valid_corners = [
                    self.queue[i][1].court_corners
                    for i in win_indices
                    if self.queue[i][1].court_corners is not None
                ]
                if len(valid_corners) >= 2:
                    smoothed_c = np.mean(valid_corners, axis=0).astype(np.float32)
                    target_res.court_corners = smoothed_c
                    target_res.homography_matrix, _ = cv2.findHomography(
                        smoothed_c, self.base_dst_points
                    )

            # Smooth tail player bounding boxes
            for tp in target_res.players:
                if tp.track_id == -1 or tp.confidence == 0.0:
                    continue

                boxes = []
                for i in win_indices:
                    match = next((p for p in self.queue[i][1].players if p.track_id == tp.track_id), None)
                    if match is not None:
                        boxes.append([match.bbox.x1, match.bbox.y1, match.bbox.x2, match.bbox.y2])

                if len(boxes) >= 2:
                    avg_box = np.mean(boxes, axis=0)
                    tp.bbox.x1, tp.bbox.y1, tp.bbox.x2, tp.bbox.y2 = avg_box
                    tp.src_coords = tp.bbox.bottom_center

    def drain(self):
        """Flushes remaining frames at end-of-stream with bounded tail smoothing."""
        if len(self.queue) == 0:
            return
    
        # Smooth only the un-processed tail frames BEFORE popping
        self._smooth_tail_frames()
    
        # Pop and project remaining frame
        while len(self.queue) > 0:
            frame, result = self.queue.popleft()
            self._project_frame_to_bev(result)
            yield frame, result

    def push(self, frame, result):
        """Appends frame and repairs history over the sliding window."""
        self.queue.append((frame, result))

        if len(self.queue) >= 5:
            self._interpolate_camera_gaps()
            self._smooth_camera_trajectory()
            self._interpolate_missing_players()
            self._smooth_player_trajectories()

    def pop(self):
        """Yields the fully stabilized, finalized frame (N - buffer_size)."""
        if len(self.queue) < self.buffer_size:
            return None, None

        frame, result = self.queue.popleft()

        # Execute BEV coordinate projection at the exit point
        self._project_frame_to_bev(result)

        return frame, result
