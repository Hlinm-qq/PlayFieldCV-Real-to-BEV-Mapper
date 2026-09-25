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
        """Called every frame"""
        if self.old_gray is None or self.tracked_features is None or len(self.tracked_features) == 0:
            return None, False
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Run standard optical flow
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, current_gray, self.tracked_features, None, **self.lk_params
        )

        good_new = new_features[status == 1]
        good_old = self.tracked_features[status == 1]

        # MASK VALIDATION: Drop any points that walked onto players
        validated_new, validated_old = [], []
        height, width = safe_floor_mask.shape
        
        for new_pt, old_pt in zip(good_new, good_old):
            x, y = int(new_pt[0]), int(new_pt[1])
            # Check bounds, then check if the mask pixel is white (> 0)
            if 0 <= y < height and 0 <= x < width:
                if safe_floor_mask[y, x] > 0:  
                    validated_new.append(new_pt)
                    validated_old.append(old_pt)

        validated_new = np.array(validated_new, dtype=np.float32).reshape(-1, 1, 2)
        validated_old = np.array(validated_old, dtype=np.float32).reshape(-1, 1, 2)

        # Calculate LK Health
        lk_health = len(validated_new) / max(1, len(self.tracked_features))
        if lk_health < 0.30 or len(validated_new) < 10:
            return None, lk_health 

        # Calculate camera delta and update court corners
        delta_H, _ = cv2.findHomography(validated_old, validated_new, cv2.RANSAC, 2.0)

        if delta_H is None:
            return None, False

        # Apply the camera movement to 4 invisible court corners
        self.corner_pixels = cv2.perspectiveTransform(self.corner_pixels, delta_H)
        # Calculate the final matrix using the newly moved corners and the 2D map
        current_H, _ = cv2.findHomography(self.corner_pixels, self.base_dst_points)

        self.old_gray = current_gray.copy()
        self.tracked_features = cv2.goodFeaturesToTrack(
            self.old_gray, mask=safe_floor_mask, **self.feature_params
        )

        if self.tracked_features is not None and len(self.tracked_features) > 0:
            term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            self.tracked_features = cv2.cornerSubPix(self.old_gray, self.tracked_features, (5, 5), (-1, -1), term_crit)
        else:
            self.tracked_features = None

        return current_H, lk_health

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
    def __init__(self):
        self.court_hist = None
        # Use an elliptical kernel for morphological cleanup
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def initialize(self, frame, court_corners):
        """Builds the statistical color profile of the court on Frame 1."""
        # Convert to HSV space to handle shadows and lighting
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a blank mask, then draw the source 4 points as a white polygon
        roi_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        corners_int = np.int32(court_corners).reshape((-1, 1, 2))
        cv2.fillPoly(roi_mask, [corners_int], 255)
        
        # Calculate a 2D histogram using only Hue (color) and Saturation (intensity)
        self.court_hist = cv2.calcHist(
            [hsv_frame], [0, 1], roi_mask, 
            [16, 16], [0, 180, 0, 256]
        )
        
        # Normalize to 0-255 for backprojection math
        cv2.normalize(self.court_hist, self.court_hist, 0, 255, cv2.NORM_MINMAX)

    def get_mask(self, frame, player_detection=None):
        """Generates a binary mask where valid area is white, and the rest are black."""
        # Base Mask Generation (Color Histogram)
        if self.court_hist is None:
            mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        else:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Backproject: OpenCV assigns a probability (0-255) to every pixel 
            back_proj = cv2.calcBackProject(
                [hsv_frame], [0, 1], self.court_hist, 
                [0, 180, 0, 256], 1
            )
            # Threshold: Only keep pixels that are highly likely to be the floor
            _, mask = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)
            # Morphological Opening: Erase isolated noisy pixels (like a shoe that matched the floor color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            # Morphological Closing: Fill small holes in the floor
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Negative Masking (Erase Players)
        if player_detection:
            for det in player_detection:
                x1, y1, x2, y2 = det["xyxy"]
                
                # Padding: Expand box by 10% to catch bleeding edges
                pad_x = int((x2 - x1) * 0.1) 
                pad_y = int((y2 - y1) * 0.1) 
                
                # Image Boundaries
                px1 = max(0, int(x1 - pad_x))
                py1 = max(0, int(y1 - pad_y))
                px2 = min(frame.shape[1], int(x2 + pad_x))
                py2 = min(frame.shape[0], int(y2 + pad_y))
                
                # Draw black rectangle over bounding box
                cv2.rectangle(mask, (px1, py1), (px2, py2), 0, -1)
                
        return mask

    def recalibrate(self, frame, new_corners):
        """Retrain the color profile using the newly discovered court area"""
        self.initialize(frame, new_corners)
