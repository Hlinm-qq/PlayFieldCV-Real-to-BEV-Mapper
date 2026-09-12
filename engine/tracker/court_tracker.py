import cv2
import numpy as np

from engine.tracker.lk_tracker import *
from engine.detection.hough_detector import *

class CourtTracker:
    def __init__(self, ema_alpha=0.15, hough_interval=60):
        # Hyperparameters
        self.ema_alpha = ema_alpha          # Smoothing factor (0.0 to 1.0)
        self.hough_interval = hough_interval

        # Specialized engines
        self.masker = DynamicCourtMasker()
        self.lk = LKTracker()
        self.hough = HoughDetector()

        # State encapsulation
        self.current_matrix = None

    def initialize(self, frame, src_corners, base_dst_points):
        # Train the color histogram on Frame 1
        self.masker.initialize(frame, src_corners)
        initial_mask = self.masker.get_mask(frame)
        self.lk.initialize(frame, src_corners, base_dst_points, initial_mask)
        
        self.current_matrix, _ = cv2.findHomography(src_corners, base_dst_points)

    def update(self, frame, frame_idx):
        if self.current_matrix is None: 
            return None

        new_matrix = None
        lk_health = 1.0

        # Generate the pristine floor mask using histogram backprojection
        safe_floor_mask = self.masker.get_mask(frame)

        # Pass the frame AND the mask to the optical flow engine
        new_matrix, lk_health = self.lk.update_matrix(frame, safe_floor_mask)

        # Fallback to Hough if LK health is low (camera blurred, etc.)
        if lk_health < 0.50 or frame_idx % self.hough_interval == 0:
            # print(f"Hough calibrated")
            hough_matrix, hough_corners = self.hough.detect(frame, self.current_matrix)

            if self._is_matrix_valid(hough_matrix):
                new_matrix = hough_matrix
                # Reset LK points to the new Hough lines
                self.lk.reset_points(frame, hough_corners, safe_floor_mask) 

        # If LK failed and Hough failed, freeze the matrix
        if not self._is_matrix_valid(new_matrix):
            return self.current_matrix, safe_floor_mask
        
        self.current_matrix = self._apply_ema(new_matrix)
        return self.current_matrix, safe_floor_mask

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

    def debug_draw_lk_mask(self, frame, mask):
        if mask is not None:
            # Show the raw black-and-white mask
            cv2.imshow("Debug: Floor Mask", mask)
            
            # Create green overlay on the original frame
            green_overlay = np.zeros_like(frame)
            green_overlay[:, :] = (0, 255, 0)
            
            # Bitwise AND only applies green where the mask is white
            masked_green = cv2.bitwise_and(green_overlay, green_overlay, mask=mask)
            
            # Blend with the original frame
            debug_view = cv2.addWeighted(frame, 1.0, masked_green, 0.3, 0)
            cv2.imshow("Debug: Mask Overlay", debug_view)