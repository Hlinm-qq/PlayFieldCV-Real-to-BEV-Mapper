import cv2
import numpy as np

class CameraTracker:
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