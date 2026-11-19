import numpy as np
import cv2

class PointPicker:
    def __init__(self):
        self.points = []
        self.window_name = "Calibration: Click 4 Corners"

    def click_event(self, event, x, y, flags, param):
        """Records the coordinates when the left mouse button is clicked."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                # Draw a red dot to show clicked points
                cv2.circle(self.display_frame, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow(self.window_name, self.display_frame)
                
                print(f"Point {len(self.points)} saved: [{x}, {y}]")

    def get_points(self, first_frame):
        """Opens a window, lets the user click 4 points, and returns them."""
        self.display_frame = first_frame.copy()
        self.points = [] # Reset points just in case
        
        print("\n--- CALIBRATION REQUIRED ---")
        print("Click the 4 corners of the court in clockwise order:")
        print("1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")
        
        cv2.imshow(self.window_name, self.display_frame)
        cv2.setMouseCallback(self.window_name, self.click_event)
        
        # Keep the window open until 4 points are clicked
        while len(self.points) < 4:
            cv2.waitKey(10) # Briefly pause to let OpenCV process the clicks
            
        print("4 points collected\n")
        
        # Close the calibration window and return the points
        cv2.destroyWindow(self.window_name)
        return np.float32(self.points)