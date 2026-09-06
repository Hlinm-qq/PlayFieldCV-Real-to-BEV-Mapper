import cv2
import numpy as np

class LKTracker:
    def __init__(self):
        # The configuration for the Lucas-Kanade algorithm
        # winSize: The size of the search window (15x15 pixels)
        # maxLevel: Uses an image pyramid to handle larger, faster camera movements
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Memory variables
        self.old_gray = None
        self.old_points = None
        self.base_dst_points = None

    def initialize(self, frame, src_points, dst_points):
        """
        Recalibrate LK based on current reference state
        """
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Ensure points are float32, which LK requires
        self.old_points = src_points.astype(np.float32).reshape(-1, 1, 2)
        self.base_dst_points = dst_points

    def update_matrix(self, current_frame):
        """
        Per frame updater to track the pixels and generate a Homography matrix
        """
        if self.old_gray is None or self.old_points is None:
            return None, False

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Run LUCAS-KANADE (LK)
        # Returns the new (x,y) locations of tracked points, and a 'status' array 
        # (status = 1 if it successfully tracked the point, 0 if it lost it)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.old_gray, 
            current_gray, 
            self.old_points, 
            None, 
            **self.lk_params
        )

        # Keep only the points that LK successfully found in the new frame
        good_new = new_points[status == 1]
        
        # Need at least 4 points to calculate Homography
        if len(good_new) >= 4:
            # Reformat the array for findHomography
            good_new = good_new.reshape(-1, 1, 2)
            
            # Recalculate homography matrix
            current_H, _ = cv2.findHomography(good_new, self.base_dst_points)

            # Update memory for the next frame
            self.old_gray = current_gray.copy()
            self.old_points = good_new

            return current_H, True
            
        else:
            # LK lost the points
            return None, False

    def debug_draw_lk_tracker(self, frame, lk_points):
        """
        Draws the active Lucas-Kanade tracking points and their 15x15 search windows.
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
        
        # Create a blank mask, then draw the user's 4 points as a white polygon
        roi_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        corners_int = np.int32(court_corners).reshape((-1, 1, 2))
        cv2.fillPoly(roi_mask, [corners_int], 255)
        
        # Calculate a 2D histogram using only Hue (color) and Saturation (intensity)
        # We drop the Value channel so the tracker ignores shadows
        self.court_hist = cv2.calcHist(
            [hsv_frame], [0, 1], roi_mask, 
            [16, 16], [0, 180, 0, 256]
        )
        
        # Normalize to 0-255 for backprojection math
        cv2.normalize(self.court_hist, self.court_hist, 0, 255, cv2.NORM_MINMAX)

    def get_mask(self, frame):
        """Generates a binary mask where the court is white, and players are black."""
        if self.court_hist is None:
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Backproject: OpenCV assigns a probability (0-255) to every pixel 
        # based on how closely it matches the Frame 1 histogram
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
        
        return mask

