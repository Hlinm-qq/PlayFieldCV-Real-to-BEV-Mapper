import cv2
import numpy as np

class LKTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=25, blockSize=7)
        self.old_gray = None
        self.tracked_features = None
        self.corner_pixels = None     # The 4 main court corners in pixel space
        self.base_dst_points = None   # The 2D bird's-eye view coordinates

    def initialize(self, frame, src_corners, dst_points, initial_mask):
        """Called at Frame 1"""
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corner_pixels = src_corners.astype(np.float32).reshape(-1, 1, 2)
        self.base_dst_points = dst_points
        
        # Find 100 strong points to track instead of just the 4 corners
        self.tracked_features = cv2.goodFeaturesToTrack(self.old_gray, mask=initial_mask, **self.feature_params)

        term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.tracked_features = cv2.cornerSubPix(self.old_gray, self.tracked_features, (5, 5), (-1, -1), term_crit)

    def reset_points(self, frame, new_hough_corners, safe_floor_mask):
        """Called when Hough takes over to fix the tracker."""
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Overwrite the drifted corners with the newly detected Hough corners
        self.corner_pixels = new_hough_corners.astype(np.float32).reshape(-1, 1, 2)

        # Respawn the tracking points based on the new reality
        self.tracked_features = cv2.goodFeaturesToTrack(
            self.old_gray, mask=safe_floor_mask, **self.feature_params
        )

        if self.tracked_features is not None and len(self.tracked_features) > 0:
            term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            self.tracked_features = cv2.cornerSubPix(self.old_gray, self.tracked_features, (5, 5), (-1, -1), term_crit)
        else:
            print("tracked_features invalid, Failed to reset LK")
            self.tracked_features = None

    def update_matrix(self, current_frame, safe_floor_mask):
        if self.old_gray is None or self.tracked_features is None or len(self.tracked_features) == 0:
            return None, 0.0
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Run optical flow
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, current_gray, self.tracked_features, None, **self.lk_params
        )

        good_new = new_features[status == 1]
        good_old = self.tracked_features[status == 1]

        # Mask Validation
        validated_new, validated_old = [], []
        if safe_floor_mask is not None:
            height, width = safe_floor_mask.shape
            for new_pt, old_pt in zip(good_new, good_old):
                x, y = int(new_pt[0]), int(new_pt[1])
                # Check bounds, then check if the mask pixel is white (> 0)
                if 0 <= y < height and 0 <= x < width:
                    if safe_floor_mask[y, x] > 0:  
                        validated_new.append(new_pt)
                        validated_old.append(old_pt)
        else:
            validated_new = good_new
            validated_old = good_old

        validated_new = np.array(validated_new, dtype=np.float32).reshape(-1, 2)
        validated_old = np.array(validated_old, dtype=np.float32).reshape(-1, 2)

        if len(validated_new) < 5:
            return None, 0.0 

        # Translation & Outlier Rejection
        flow_vectors = validated_new - validated_old
        median_translation = np.median(flow_vectors, axis=0) # [dx, dy]

        # Strip out points that aren't moving with the dominant floor
        deviations = np.linalg.norm(flow_vectors - median_translation, axis=1)
        inlier_mask = deviations < 4.0
        
        inliers_new = validated_new[inlier_mask]
        inliers_old = validated_old[inlier_mask]
        
        lk_health = len(inliers_new) / max(1, len(self.tracked_features))
        if len(inliers_new) < 5 or lk_health < 0.25:
            return None, lk_health

        final_translation = np.mean(inliers_new - inliers_old, axis=0)

        # Create a Rigid Translation Matrix 
        delta_H = np.array([
            [1.0, 0.0, final_translation[0]],
            [0.0, 1.0, final_translation[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Update corner pixels
        new_corner_pixels = cv2.perspectiveTransform(self.corner_pixels.reshape(-1, 1, 2), delta_H)

        # Prepare for next frame
        self.old_gray = current_gray.copy()
        self.tracked_features = cv2.goodFeaturesToTrack(
            self.old_gray, mask=safe_floor_mask, **self.feature_params
        )
        if self.tracked_features is not None and len(self.tracked_features) > 0:
            term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            self.tracked_features = cv2.cornerSubPix(self.old_gray, self.tracked_features, (5, 5), (-1, -1), term_crit)

        return new_corner_pixels, lk_health

    def debug_draw_lk_tracker(self, frame, lk_points):
        """
        Draws the active Lucas-Kanade tracking points.
        """
        debug_frame = frame.copy()
        
        if lk_points is not None:
            for pt in lk_points:
                # Extract the x and y coordinates
                x, y = int(pt[0][0]), int(pt[0][1])
                
                # Draw the center point
                cv2.circle(debug_frame, (x, y), 4, (255, 0, 255), -1)
                
                # Draw the search window LK is using
                cv2.rectangle(debug_frame, (x - 7, y - 7), (x + 7, y + 7), (0, 255, 0), 1)
                
        cv2.imshow("Debug: LK Tracker", debug_frame)
        cv2.waitKey(1)
        
        return debug_frame

class DynamicCourtMasker:
    def __init__(self, court_length, court_width, court_margin=2):
        """
        court_margin: Distance outside boundary lines (in metric court units)
                         to include as valid playable floor (free zone).
        """
        self.court_length = court_length
        self.court_width = court_width
        self.court_margin = court_margin
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def initialize(self, frame, court_corners, player_detections=None):
        # Verification check
        return court_corners is not None and len(court_corners) == 4

    def recalibrate(self, frame=None, new_corners=None, player_detections=None):
        # The mask updates automatically from the latest tracked corners.
        pass

    def get_mask(self, frame, court_corners=None, current_H=None, base_dst_points=None, player_detections=None):
        """
        Generates a floor mask by combining:
        1. Projected geometric court boundaries
        2. Bounded free-zone expansion
        3. Detected player subtractions
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # GEOMETRIC FLOOR REGION
        # If homography matrix is available, compute expanded polygon
        if current_H is not None and base_dst_points is not None:
            try:
                h_inv = np.linalg.inv(current_H)
                
                # Expand court bounds by court_margin
                m = self.court_margin
                expanded_dst = np.array([
                    [-m, -m],
                    [self.court_length + m, -m],
                    [self.court_length + m, self.court_width + m],
                    [-m, self.court_width + m]
                ], dtype=np.float32).reshape(-1, 1, 2)

                expanded_corners = cv2.perspectiveTransform(expanded_dst, h_inv)
                corners_int = np.int32(expanded_corners).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [corners_int], 255)
            except np.linalg.LinAlgError:
                # Fallback to direct corners if matrix inversion fails
                if court_corners is not None:
                    corners_int = np.int32(court_corners).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [corners_int], 255)
        elif court_corners is not None:
            corners_int = np.int32(court_corners).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [corners_int], 255)
        else:
            # Cold start fallback
            mask[:] = 255

        # CLIP TO VISIBLE FRAME BOUNDARIES
        # Prevents negative coordinates or off-screen projections from distorting OpenCV ops
        cv2.rectangle(mask, (0, 0), (w - 1, h - 1), 255, 1)

        # PLAYER SUBTRACTION (Negative Masking)
        if player_detections:
            for det in player_detections:
                x1, y1, x2, y2 = det["xyxy"]
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)

                px1 = max(0, int(x1 - pad_x))
                py1 = max(0, int(y1 - pad_y))
                px2 = min(w, int(x2 + pad_x))
                py2 = min(h, int(y2 + pad_y))

                cv2.rectangle(mask, (px1, py1), (px2, py2), 0, -1)

        return mask