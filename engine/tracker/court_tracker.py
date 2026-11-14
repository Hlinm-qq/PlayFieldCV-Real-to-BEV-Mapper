import cv2
import numpy as np

from engine.tracker.lk_tracker import *
from engine.detection.hough_detector import *
from utils.utils import *
from data_manager.output_manager import PipelineDebugger

class CornerSmoother:
    def __init__(self, max_step_px=10.0, alpha_lk=0.80, alpha_hough=0.35):
        """
        max_step_px: Maximum allowed movement per frame in pixels (prevents jumping).
        alpha_lk: High responsiveness for continuous optical flow.
        alpha_hough: Moderate smoothing when Hough recalibrates (blends changes smoothly).
        """
        self.corners = None
        self.max_step_px = max_step_px
        self.alpha_lk = alpha_lk
        self.alpha_hough = alpha_hough

    def initialize(self, initial_corners):
        self.corners = initial_corners.astype(np.float32).copy()

    def update(self, target_corners, source="lk"):
        if self.corners is None:
            self.corners = target_corners.astype(np.float32).copy()
            return self.corners

        target = target_corners.astype(np.float32)
        alpha = self.alpha_hough if source == "hough" else self.alpha_lk

        # Calculate displacement vector for each of the 4 corners
        delta = target - self.corners  # shape: (4, 1, 2)
        distances = np.linalg.norm(delta, axis=2)  # shape: (4, 1)
        max_dist = np.max(distances)

        # Slew-rate limiter: if Hough or LK jumps by more than max_step_px, clamp the step
        if max_dist > self.max_step_px:
            clamp_factor = self.max_step_px / max_dist
            delta = delta * clamp_factor

        # Apply exponential blend
        self.corners = self.corners + (alpha * delta)
        return self.corners.copy()
    
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

        self.lk_failure_streak = 0
        self.smoother = CornerSmoother(max_step_px=10.0, alpha_lk=0.85, alpha_hough=0.30)

    def initialize(self, frame, src_corners, base_dst_points):
        self.base_dst_points = base_dst_points.astype(np.float32).reshape(-1, 1, 2)
        self.current_corners = src_corners.astype(np.float32).reshape(-1, 1, 2)
                
        self.current_matrix, _ = cv2.findHomography(self.current_corners, self.base_dst_points)

        # Train the color histogram on Frame 1
        self.masker.initialize(frame, src_corners)
        initial_mask = self.masker.get_mask(frame, self.current_corners, self.current_matrix, self.base_dst_points)
        self.lk.initialize(frame, src_corners, base_dst_points, initial_mask)

        self.smoother.initialize(self.current_corners)
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
        
        health_thres = 0.40

        if lk_health < health_thres or new_corners is None:
            self.lk_failure_streak += 1
        else:
            self.lk_failure_streak = 0

        needs_hough = (self.lk_failure_streak >= 4) or (frame_idx % self.hough_interval == 0)
        hough_applied = False
        detected_corners, dst_pts = None, None
        
        # Fallback to Hough if LK health is low or when recalibration
        # Hough Verification / Recalibration
        if needs_hough:
            h_matrix, h_corners, h_dst = self.hough.detect(frame, self.current_matrix)

            # Validate reprojection error
            if h_matrix is not None and self._validate_hough_reprojection(h_corners, h_dst):
                # SMOOTHER UPDATE: Source = "hough" (smooths reference switches)
                self.current_corners = self.smoother.update(h_corners, source="hough")

                # Synchronize Masker and LK with the smoothed corners
                self.masker.recalibrate(frame, new_corners=self.current_corners)
                new_mask = self.masker.get_mask(frame)
                self.lk.reset_points(frame, self.current_corners, new_mask)

                detected_corners = h_corners
                dst_pts = h_dst
                self.lk_failure_streak = 0
                hough_applied = True

        # LK Flow Continuous Update (if Hough did not run or failed)
        if not hough_applied:
            if new_corners is not None and lk_health >= 0.20:
                # SMOOTHER UPDATE: Source = "lk" (fluid frame-to-frame tracking)
                self.current_corners = self.smoother.update(new_corners, source="lk")
            else:
                # Both failed: Coast on the previous smoothed position
                pass

        # Keep LK tracking anchor and Homography matrix aligned to active state
        self.lk.corner_pixels = self.current_corners.copy()
        self.current_matrix, _ = cv2.findHomography(self.current_corners, self.base_dst_points)

        return self.current_matrix, safe_floor_mask, detected_corners, dst_pts

    # def _is_matrix_valid(self, matrix):
    #     """Mathematical guardrail to prevent tracking collapse"""
    #     if matrix is None or matrix.shape != (3, 3):
    #         return False
            
    #     # If the determinant of the top-left 2x2 is near zero, 
    #     # the perspective has collapsed into a 1D line.
    #     det = np.linalg.det(matrix[0:2, 0:2])
    #     if abs(det) < 1e-4:
    #         return False
            
    #     return True

    # def _apply_ema(self, new_matrix):
    #     """Prevents the 2D court overlay from vibrating frame-to-frame"""
    #     # Normalize matrices so H[2,2] is 1.0 before blending
    #     new_matrix = new_matrix / new_matrix[2, 2]
    #     old_matrix = self.current_matrix / self.current_matrix[2, 2]
        
    #     smoothed = (self.ema_alpha * new_matrix) + ((1 - self.ema_alpha) * old_matrix)
    #     return smoothed / smoothed[2, 2]

    # def _validate_hough_reprojection(self, hough_corners, dst_pts, max_error_units=60.0):
    #     """
    #     Validates Hough detection across variable court regions by projecting
    #     image detections into canonical court coordinates via the current matrix.
    #     """
    #     if hough_corners is None or dst_pts is None or self.current_matrix is None:
    #         return False

    #     h_pts = hough_corners.astype(np.float32).reshape(-1, 1, 2)
    #     target_pts = dst_pts.astype(np.float32).reshape(-1, 2)

    #     # Map detected image points into court space using the active camera tracking state
    #     projected_court_pts = cv2.perspectiveTransform(h_pts, self.current_matrix).reshape(-1, 2)

    #     # Calculate Euclidean error in court metric units
    #     point_errors = np.linalg.norm(projected_court_pts - target_pts, axis=1)
    #     mean_error = np.mean(point_errors)

    #     # Reject if average reprojection deviation exceeds threshold
    #     return mean_error < max_error_units

    def _validate_hough_reprojection(self, hough_corners, dst_pts, max_single_error=15.0, max_mean_error=8.0):
        """
        Validates detected intersections against the expected court layout.
        - max_single_error: Max allowed error for ANY single point (15.0 px = 15 cm).
        - max_mean_error: Max allowed average error across all points (8.0 px = 8 cm).
        """
        if hough_corners is None or dst_pts is None or self.current_matrix is None:
            return False

        h_pts = hough_corners.astype(np.float32).reshape(-1, 1, 2)
        target_pts = dst_pts.astype(np.float32).reshape(-1, 2)

        # Map detected camera intersections into canonical court space
        projected_court_pts = cv2.perspectiveTransform(h_pts, self.current_matrix).reshape(-1, 2)

        # Euclidean distance per point in canvas pixels (100 px = 1 meter)
        point_errors = np.linalg.norm(projected_court_pts - target_pts, axis=1)

        # Reject if ANY single intersection deviates by more than threshold
        if np.max(point_errors) > max_single_error:
            return False

        # Reject if overall fit deviation exceeds mean error
        if np.mean(point_errors) > max_mean_error:
            return False

        # Geometric sanity check: verify court lines remain roughly perpendicular
        if len(hough_corners) >= 4:
            v0 = hough_corners[1][0] - hough_corners[0][0] # Top edge vector
            v1 = hough_corners[3][0] - hough_corners[0][0] # Left edge vector
            cos_angle = np.abs(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-6))
            # In perspective, angle should not collapse (cos_angle should stay well below 0.35)
            if cos_angle > 0.35:
                return False

        return True

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