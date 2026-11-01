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
            return None, 0.0
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Run standard optical flow
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, current_gray, self.tracked_features, None, **self.lk_params
        )

        good_new = new_features[status == 1]
        good_old = self.tracked_features[status == 1]

        # MASK VALIDATION: Drop any points that walked onto players
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
        
        validated_new = np.array(validated_new, dtype=np.float32).reshape(-1, 1, 2)
        validated_old = np.array(validated_old, dtype=np.float32).reshape(-1, 1, 2)

        # Calculate LK Health
        lk_health = len(validated_new) / max(1, len(self.tracked_features))
        if lk_health < 0.30 or len(validated_new) < 10:
            return None, lk_health 

        # Estimate Rigid/Affine motion (translation + rotation + scale)
        # Prevents the wild perspective warping of an 8-DOF homography on inter-frame steps
        affine_transform, inliers = cv2.estimateAffinePartial2D(
            validated_old, validated_new, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )

        if affine_transform is None:
            return None, lk_health

        # Convert 2x3 affine matrix to 3x3 homogeneous transformation
        delta_H = np.eye(3, dtype=np.float32)
        delta_H[0:2, :] = affine_transform

        # Update corner pixels
        new_corner_pixels = cv2.perspectiveTransform(self.corner_pixels, delta_H)        
        # delta_H, _ = cv2.findHomography(validated_old, validated_new, cv2.RANSAC, 2.0)

        # if delta_H is None:
        #     return None, 0.0

        # # Apply the camera movement to 4 invisible court corners
        # self.corner_pixels = cv2.perspectiveTransform(self.corner_pixels, delta_H)
        # # Calculate the final matrix using the newly moved corners and the 2D map
        # current_H, _ = cv2.findHomography(self.corner_pixels, self.base_dst_points)

        # Prepare for next frame
        self.old_gray = current_gray.copy()
        self.tracked_features = cv2.goodFeaturesToTrack(
            self.old_gray, mask=safe_floor_mask, **self.feature_params
        )

        if self.tracked_features is not None and len(self.tracked_features) > 0:
            term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            self.tracked_features = cv2.cornerSubPix(self.old_gray, self.tracked_features, (5, 5), (-1, -1), term_crit)
        else:
            self.tracked_features = None

        # return current_H, lk_health
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
    def __init__(self, blend_alpha=0.10):
        self.court_hist = None
        # Use an elliptical kernel for morphological cleanup
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Structuring element to erode the boundary inward before sampling
        self.inset_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        self.blend_alpha = blend_alpha

    def initialize(self, frame, court_corners, player_detections=None):
        """Builds the statistical color profile"""
        # Convert to HSV space to handle shadows and lighting
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi_mask = self._build_sampling_roi(frame.shape, court_corners, player_detections)
        
        if cv2.countNonZero(roi_mask) < 5000:
            return False

        hist = cv2.calcHist(
            [hsv_frame], [0, 1], roi_mask, 
            [16, 16], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self.court_hist = hist
        return True

    def _build_sampling_roi(self, shape, court_corners, player_detections=None):
        """
        Creates an ROI mask restricted to the interior floor,
        to exclude players and boundary lines
        """
        roi_mask = np.zeros(shape[:2], dtype=np.uint8)
        corners_int = np.int32(court_corners).reshape((-1, 1, 2))
        cv2.fillPoly(roi_mask, [corners_int], 255)
        
        # cv2.imshow("asdasdasd", roi_mask)
        # print("corners_int")
        # print(corners_int)

        # Inward Erosion: Pull sampling boundary away from lines and edges
        # roi_mask = cv2.erode(roi_mask, self.inset_kernel, iterations=2)
        # cv2.imshow("123123123", roi_mask)

        # Negative Masking: Subtract all detected player bounding boxes
        if player_detections:
            for det in player_detections:
                x1, y1, x2, y2 = det["xyxy"]
                # 20% padding to handle noise bleeding
                pad_x = int((x2 - x1) * 0.10)
                pad_y = int((y2 - y1) * 0.10)

                px1 = max(0, int(x1 - pad_x))
                py1 = max(0, int(y1 - pad_y))
                px2 = min(shape[1], int(x2 + pad_x))
                py2 = min(shape[0], int(y2 + pad_y))

                cv2.rectangle(roi_mask, (px1, py1), (px2, py2), 0, -1)

        return roi_mask

    def get_mask(self, frame, player_detection=None):
        """Generates a binary mask where valid area is white, and the rest are black."""
        # Base Mask Generation (Color Histogram)
        if self.court_hist is None:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("asdasdasd2", hsv_frame)
        # Backproject: assigns a probability (0-255) to every pixel 
        back_proj = cv2.calcBackProject(
            [hsv_frame], [0, 1], self.court_hist, 
            [0, 180, 0, 256], 1
        )
        cv2.imshow("1234124123", back_proj)
        # Threshold: Only keep pixels that are highly likely to be the floor
        _, mask = cv2.threshold(back_proj, 10, 255, cv2.THRESH_BINARY)
        # Morphological Opening: Erase isolated noisy pixels
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

    def recalibrate(self, frame, new_corners, player_detections=None):
        """
        Soft-updates the color profile using Bhattacharyya rejection gating.
        Rejects candidates contaminated by occlusions
        """
        if self.court_hist is None:
            return self.initialize(frame, new_corners, player_detections)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi_mask = self._build_sampling_roi(frame.shape, new_corners, player_detections)

        if cv2.countNonZero(roi_mask) < 5000:
            return 

        candidate_hist = cv2.calcHist(
            [hsv_frame], [0, 1], roi_mask, 
            [16, 16], [0, 180, 0, 256]
        )
        cv2.normalize(candidate_hist, candidate_hist, 0, 255, cv2.NORM_MINMAX)

        # Compare candidate with master histogram 
        # (0.0 = identical, 1.0 = completely disjoint)
        bhattacharyya_dist = cv2.compareHist(
            self.court_hist, candidate_hist, cv2.HISTCMP_BHATTACHARYYA
        )

        # REJECTION GATE
        if bhattacharyya_dist > 0.35:
            return

        # Complementary blend: absorb subtle lighting changes without destabilizing the model
        blended = (1.0 - self.blend_alpha) * self.court_hist + (self.blend_alpha * candidate_hist)
        cv2.normalize(blended, self.court_hist, 0, 255, cv2.NORM_MINMAX)
    
