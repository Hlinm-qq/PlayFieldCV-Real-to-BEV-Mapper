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

        delta_H = None
        is_homography_valid = False

        # Attempt Full 8-DOF Projective Homography
        if len(validated_new) >= 16:
            H_cand, inliers = cv2.findHomography(
                validated_old, validated_new, method=cv2.RANSAC, ransacReprojThreshold=2.0
            )
            if H_cand is not None:
                # Check condition number to avoid singular/collapsing matrices
                cond = np.linalg.cond(H_cand)
                # Perspective terms h20 and h21 must represent plausible frame-to-frame angular velocity
                h20, h21 = abs(H_cand[2, 0]), abs(H_cand[2, 1])
                
                # Well-behaved inter-frame homographies have cond < 50 and tiny bottom-row values
                if cond < 50.0 and h20 < 1e-3 and h21 < 1e-3:
                    delta_H = H_cand
                    is_homography_valid = True

        # Estimate Affine Transform (Translation + Rotation + Scale)
        # Fallback to Partial Affine if Homography fails or floor points are poorly distributed
        if not is_homography_valid:
            affine_mat, inliers = cv2.estimateAffinePartial2D(
                validated_old, validated_new, method=cv2.RANSAC, ransacReprojThreshold=2.5
            )
            if affine_mat is None:
                return None, 0.0
            delta_H = np.eye(3, dtype=np.float32)
            delta_H[0:2, :] = affine_mat

        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        lk_health = inlier_count / max(1, len(self.tracked_features))

        if inlier_count < 6 or lk_health < 0.20:
            return None, lk_health

        # # Lift 2x3 affine matrix to 3x3 projective matrix
        # delta_H = np.eye(3, dtype=np.float32)
        # delta_H[0:2, :] = affine_matrix

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
    def __init__(self, court_length=1800, court_width=900, court_margin_meters=0.5, px_per_meter=100.0):
        self.court_length = int(court_length)
        self.court_width = int(court_width)
        
        self.margin_px = int(court_margin_meters * px_per_meter)

        # 4 corners in canonical court coordinates (expanded outwards by margin)
        m = self.margin_px
        self.canonical_floor_corners = np.array([
            [-m, -m],                                  # Top-Left
            [self.court_length + m, -m],               # Top-Right
            [self.court_length + m, self.court_width + m], # Bottom-Right
            [-m, self.court_width + m]                 # Bottom-Left
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Persistent internal state
        self.cached_corners = None
        self.cached_H = None
        self.base_dst_points = np.array([
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width],
            [0, self.court_width]
        ], dtype=np.float32).reshape(-1, 1, 2)

    def initialize(self, frame, court_corners, initial_H=None):
        self.cached_corners = court_corners.copy() if court_corners is not None else None
        if initial_H is not None:
            self.cached_H = initial_H.copy()
        elif self.cached_corners is not None:
            self.cached_H, _ = cv2.findHomography(self.cached_corners, self.base_dst_points)
        return self.cached_corners is not None

    def recalibrate(self, frame=None, new_corners=None, new_H=None):
        """Saves recalibrated state so zero-arg get_mask(frame) maintains court boundary."""
        if new_corners is not None:
            self.cached_corners = new_corners.copy()
        if new_H is not None:
            self.cached_H = new_H.copy()
        elif self.cached_corners is not None:
            self.cached_H, _ = cv2.findHomography(self.cached_corners, self.base_dst_points)

    def get_mask(self, frame, court_corners=None, current_H=None, base_dst_points=None, player_detections=None):
        h, w = frame.shape[:2]

        # Sync state if provided
        if current_H is not None:
            self.cached_H = current_H
        if court_corners is not None:
            self.cached_corners = court_corners

        mask = np.zeros((h, w), dtype=np.uint8)

        # GEOMETRIC FLOOR REGION
        if self.cached_H is not None:
            try:
                # Map expanded court corners from BEV space -> Image space
                h_inv = np.linalg.inv(self.cached_H)
                img_corners = cv2.perspectiveTransform(self.canonical_floor_corners, h_inv)
                corners_int = np.int32(img_corners).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [corners_int], 255)
            except np.linalg.LinAlgError:
                if self.cached_corners is not None:
                    corners_int = np.int32(self.cached_corners).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [corners_int], 255)
        elif self.cached_corners is not None:
            corners_int = np.int32(self.cached_corners).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [corners_int], 255)
        else:
            mask[:] = 255  # Cold start fallback

        # DYNAMIC NET EXCLUSION (high-value stabilization)
        # Masks out the center net band to prevent LK tracking non-floor net tape
        if self.cached_H is not None:
            self._mask_net_post_zone(mask, w, h)

        # PLAYER SUBTRACTION (Negative Masking)
        if player_detections:
            self._subtract_players(mask, player_detections, w, h)

        return mask

    def _mask_net_post_zone(self, mask, w, h):
        """Blanks a thin vertical strip around the center net to prevent tracking moving mesh/tape."""
        try:
            h_inv = np.linalg.inv(self.cached_H)
            net_x = self.court_length // 2
            # 30px width in metric court = 30cm around center line
            net_poly_bev = np.array([
                [net_x - 15, -self.margin_px],
                [net_x + 15, -self.margin_px],
                [net_x + 15, self.court_width + self.margin_px],
                [net_x - 15, self.court_width + self.margin_px]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            net_poly_img = cv2.perspectiveTransform(net_poly_bev, h_inv)
            cv2.fillPoly(mask, [np.int32(net_poly_img)], 0)
        except np.linalg.LinAlgError:
            pass

    def _subtract_players(self, mask, detections, w, h):
        """Flexible extractor handling dicts, dataclasses, and bounding box formats."""
        for det in detections:
            bbox = None
            if isinstance(det, dict):
                bbox = det.get("xyxy", None)
            elif hasattr(det, "bbox"):
                b = det.bbox
                bbox = [b.x1, b.y1, b.x2, b.y2]
            elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 4:
                bbox = det[:4]

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # 10% padding around player body
                pad_x = int((x2 - x1) * 0.10)
                pad_y = int((y2 - y1) * 0.10)

                px1 = max(0, int(x1 - pad_x))
                py1 = max(0, int(y1 - pad_y))
                px2 = min(w, int(x2 + pad_x))
                py2 = min(h, int(y2 + pad_y))

                cv2.rectangle(mask, (px1, py1), (px2, py2), 0, -1)

