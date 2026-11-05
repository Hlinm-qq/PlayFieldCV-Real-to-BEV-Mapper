import cv2
import numpy as np

from engine.tracker.lk_tracker import *
from engine.detection.hough_detector import *
from utils.utils import *
from data_manager.output_manager import PipelineDebugger

class CourtTracker:
    def __init__(
            self, 
            ema_alpha=0.15, 
            hough_interval=60, 
            debugger : PipelineDebugger = None,
            court_size : tuple = (1800, 900)
        ):
        # Frame Debugger
        self.debugger = debugger

        # Hyperparameters
        self.ema_alpha = ema_alpha          # Smoothing factor (0.0 to 1.0)
        self.hough_interval = hough_interval

        # Specialized engines
        self.masker = DynamicCourtMasker(court_length=court_size[0], court_width=court_size[1])
        self.lk = LKTracker()
        self.hough = HoughDetector(on_debug=debugger.add)

        # State encapsulation
        self.current_matrix = None

        self.current_corners = None
        self.base_dst_points = None

        self.correction_step = None       # How much to nudge per frame
        self.correction_frames_left = 0   # How many frames left to nudge

        self.lk_failure_streak = 0

    def initialize(self, frame, src_corners, base_dst_points):
        self.base_dst_points = base_dst_points.astype(np.float32).reshape(-1, 1, 2)
        self.current_corners = src_corners.astype(np.float32).reshape(-1, 1, 2)
                
        self.current_matrix, _ = cv2.findHomography(self.current_corners, self.base_dst_points)

        # Train the color histogram on Frame 1
        self.masker.initialize(frame, src_corners)
        initial_mask = self.masker.get_mask(frame, self.current_corners, self.current_matrix, self.base_dst_points)
        self.lk.initialize(frame, src_corners, base_dst_points, initial_mask)

        # cv2.imshow("initial_mask", initial_mask)
        # print("init hough result")
        # print(self.current_matrix)
        # print(self.current_corners)
        # print(self.base_dst_points)

    def update(self, frame, frame_idx, player_detection=None):
        if self.current_matrix is None: 
            return None, None, None, None

        # Generate floor mask using histogram backprojection
        safe_floor_mask = self.masker.get_mask(
            frame, self.current_corners, self.current_matrix, self.base_dst_points, player_detection
        )

        # Pass frame and the mask to optical flow engine
        new_corners, lk_health = self.lk.update_matrix(frame, safe_floor_mask)
        
        detected_corners = None
        dst_pts = None
        health_thres = 0.40

        if lk_health < health_thres or new_corners is None:
            self.lk_failure_streak += 1
        else:
            self.lk_failure_streak = 0

        needs_hough = (self.lk_failure_streak >= 4) or (frame_idx % self.hough_interval == 0)
        
        # Fallback to Hough if LK health is low or when recalibration
        if needs_hough:
            hough_matrix, hough_corners, dst_pts32 = self.hough.detect(frame, self.current_matrix)
            detected_corners = hough_corners
            dst_pts = dst_pts32

            if self._validate_hough_reprojection(hough_corners, dst_pts32, max_error_units=60.0):
                # Valid matrix mapping image -> court from the detected sub-area
                valid_h_matrix, _ = cv2.findHomography(
                    hough_corners.reshape(-1, 1, 2), 
                    dst_pts32.reshape(-1, 1, 2), 
                    cv2.RANSAC, 
                    3.0
                )

                if valid_h_matrix is not None:
                    h_inv = np.linalg.inv(valid_h_matrix)
                    canonical_hough_corners = cv2.perspectiveTransform(self.base_dst_points, h_inv)

                    if new_corners is not None and lk_health >= 0.40:
                        new_corners = (0.75 * new_corners) + (0.25 * canonical_hough_corners)
                    else:
                        new_corners = canonical_hough_corners

                    # Recalibrate masker and LK tracker
                    self.masker.recalibrate(frame, hough_corners)
                    new_mask = self.masker.get_mask(frame, new_corners, valid_h_matrix, dst_pts32, player_detection)
                    self.lk.reset_points(frame, canonical_hough_corners, new_mask)

        if new_corners is None:
            return self.current_matrix, safe_floor_mask, detected_corners, dst_pts

        self.current_corners = new_corners.copy()
        self.lk.corner_pixels = self.current_corners.copy()

        # Update homography using smoothed canonical points
        self.current_matrix, _ = cv2.findHomography(self.current_corners, self.base_dst_points)

        print("self.current_corners")
        print(self.current_corners)
        print(self.current_matrix)
        print(lk_health)
        
        return self.current_matrix, safe_floor_mask, detected_corners, dst_pts

    def _is_matrix_valid(self, matrix):
        """Mathematical guardrail to prevent tracking collapse"""
        if matrix is None or matrix.shape != (3, 3):
            return False
            
        # If the determinant of the top-left 2x2 is near zero, 
        # the perspective has collapsed into a 1D line.
        det = np.linalg.det(matrix[0:2, 0:2])
        if abs(det) < 1e-4:
            return False
            
        return True

    def _apply_ema(self, new_matrix):
        """Prevents the 2D court overlay from vibrating frame-to-frame"""
        # Normalize matrices so H[2,2] is 1.0 before blending
        new_matrix = new_matrix / new_matrix[2, 2]
        old_matrix = self.current_matrix / self.current_matrix[2, 2]
        
        smoothed = (self.ema_alpha * new_matrix) + ((1 - self.ema_alpha) * old_matrix)
        return smoothed / smoothed[2, 2]

    def _validate_hough_reprojection(self, hough_corners, dst_pts, max_error_units=60.0):
        """
        Validates Hough detection across variable court regions by projecting
        image detections into canonical court coordinates via the current matrix.
        """
        if hough_corners is None or dst_pts is None or self.current_matrix is None:
            return False

        h_pts = hough_corners.astype(np.float32).reshape(-1, 1, 2)
        target_pts = dst_pts.astype(np.float32).reshape(-1, 2)

        # Map detected image points into court space using the active camera tracking state
        projected_court_pts = cv2.perspectiveTransform(h_pts, self.current_matrix).reshape(-1, 2)

        # Calculate Euclidean error in court metric units
        point_errors = np.linalg.norm(projected_court_pts - target_pts, axis=1)
        mean_error = np.mean(point_errors)

        # Reject if average reprojection deviation exceeds threshold
        return mean_error < max_error_units

    def debug_draw_lk_mask(self, frame, mask):
        if mask is not None:
            if self.debugger is None:
                # Show the raw black-and-white mask
                cv2.imshow("Debug: Floor Mask", mask)
            
            # Create green overlay on the original frame
            green_overlay = np.zeros_like(frame)
            green_overlay[:, :] = (0, 255, 0)
            
            # Bitwise AND only applies green where the mask is white
            masked_green = cv2.bitwise_and(green_overlay, green_overlay, mask=mask)
            
            # Blend with the original frame
            debug_view = cv2.addWeighted(frame, 1.0, masked_green, 0.3, 0)

            if self.debugger is None:
                cv2.imshow("Debug: Mask Overlay", debug_view)

            combined_view = stitch_two_frames(mask, debug_view)
            return combined_view

        return None