import cv2

class CameraTracker:
    def __init__(self, dest_points):
        self.p0 = None
        self.old_gray = None
        self.dest_points = dest_points
        
        # Lucas-Kanade configuration
        self.lk_params = dict(
            winSize=(21, 21), 
            maxLevel=2, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def initialize(self, first_frame, initial_points):
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = initial_points

    def update_matrix(self, current_frame):
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Track the points from old_gray to frame_gray
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.p0, None, **self.lk_params
        )
        
        # Filter out points that Lucas-Kanade lost (status == 0)
        good_new = p1[status == 1]
        
        # Need at least 4 points to calculate a Homography matrix
        if len(good_new) < 4:
            return None, False 
            
        # Calculate the perspective transformation matrix
        H, _ = cv2.findHomography(good_new, self.dest_points)
        
        # Overwrite the old data with the new data for the NEXT frame loop
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        
        return H, True