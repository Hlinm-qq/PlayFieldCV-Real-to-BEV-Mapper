from engine.tracker.lk_tracker import *
from engine.detection.hough_detector import *

class CourtTracker:
    def __init__(self):
        self.masker = DynamicCourtMasker()
        self.lk = LKTracker()
        self.hough = HoughDetector()
        self.current_matrix = None # Current Homography matrix

    def initialize_manual(self, frame, src_corners, base_dst_points):
        # Train the color histogram on Frame 1
        self.masker.initialize(frame, src_corners)
        
        # Setup LK Tracker with 4 corners
        self.current_matrix = self.lk.initialize(frame, src_corners, base_dst_points)
        return self.current_matrix

    def update(self, frame, frame_idx):
        if self.current_matrix is None: 
            return None

        # Generate floor mask using histogram backprojection
        safe_floor_mask = self.masker.get_mask(frame)

        # Pass frame AND mask to optical flow engine
        new_matrix, lk_health = self.lk.update_matrix(frame, safe_floor_mask)

        # Fallback to Hough if LK health is low (camera blurred, etc.)
        if lk_health < 0.50:
            new_matrix = self.hough.detect(frame)
            # Reset LK points to the new Hough lines
            self.lk.reset_points(frame, safe_floor_mask) 

        self.current_matrix = self._apply_ema(new_matrix)
        return self.current_matrix

    def _apply_ema(self, new_matrix):
        """Prevents the 2D court overlay from vibrating frame-to-frame."""
        # Normalize matrices so H[2,2] is 1.0 before blending
        new_matrix = new_matrix / new_matrix[2, 2]
        old_matrix = self.current_matrix / self.current_matrix[2, 2]
        
        smoothed = (self.ema_alpha * new_matrix) + ((1 - self.ema_alpha) * old_matrix)
        return smoothed / smoothed[2, 2]

    def _is_matrix_valid(self, matrix):
        """Mathematical guardrail to prevent tracking collapse."""
        if matrix is None or matrix.shape != (3, 3):
            return False
            
        # If the determinant of the top-left 2x2 is near zero, 
        # the perspective has collapsed into a 1D line.
        det = np.linalg.det(matrix[0:2, 0:2])
        if abs(det) < 1e-4:
            return False
            
        return True