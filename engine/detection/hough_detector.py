import cv2
import numpy as np

class HoughDetector:
    def __init__(self, on_debug=None):
        self.lower_color = np.array([0, 0, 200]) 
        self.upper_color = np.array([180, 50, 255])

        self.floor_lower = np.array([10, 50, 50])   
        self.floor_upper = np.array([30, 255, 255])

        # Temporal Memory State
        self.previous_master_lines = None

        self.blueprint = {
                            "left_baseline": [[0, 0], [0, 900]],
                            "left_attackline": [[600, 0], [600, 900]],
                            "center_line": [[900, 0], [900, 900]],
                            "right_attackline": [[1200, 0], [1200, 900]],
                            "right_baseline": [[1800, 0], [1800, 900]],
                            "top_sideline": [[0, 0], [1800, 0]],
                            "bottom_sideline": [[0, 900], [1800, 900]]
                        }

        self.is_show_debug = True
        self.on_debug = on_debug

        if callable(self.on_debug):
            self.is_show_debug = False

    def _line_intersection(self, line1, line2):
        # Unpack the lines [x1, y1, x2, y2]
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Standard algebraic line intersection math (Determinant method)
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None # Lines are parallel

        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return [int(px), int(py)]

    def _convert_equations_to_endpoints(self, master_h, master_v, width, height):
        """Convert Intersect and Slope based coordinate into start and end 4 points coordinate"""
        lines = []
        # Calculate horizontal endpoints (from left edge X=0 to right edge X=width)
        for slope, intercept in master_h:
            y1 = int(intercept)
            y2 = int(slope * width + intercept)
            lines.append([[0, y1, width, y2]])
            
        # Calculate vertical endpoints (from top edge Y=0 to bottom edge Y=height)
        for slope, intercept in master_v:
            x1 = int(intercept)
            x2 = int(slope * height + intercept)
            lines.append([[x1, 0, x2, height]])
            
        return lines

    def _point_to_line_dist(self, px, py, line_p1, line_p2):
        """ Helper mathematical function to avoid divide-by-zero vertical line crashes """
        x1, y1 = line_p1
        x2, y2 = line_p2
        
        # Standard formula for perpendicular distance from a point to a line
        numerator = abs((x2 - x1)*(y1 - py) - (x1 - px)*(y2 - y1))
        denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if denominator == 0: return float('inf')
        return numerator / denominator

    def _calculate_angle_diff(self, angle1, angle2):
        """Calculates the true difference between two line angles (0 to 180 degrees)"""
        # Normalize both angles to be strictly between 0 and 180
        a1 = angle1 % 180
        a2 = angle2 % 180
        
        diff = abs(a1 - a2)
        
        # If the difference is greater than 90, it means we crossed the 0/180 boundary
        if diff > 90:
            diff = 180 - diff
            
        return diff

    def _get_line_angle(self, x1, y1, x2, y2):
        """Returns the angle of a line in degrees (0 to 180)"""
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return angle % 180

    def _calculate_intersection(self, line1, line2):
        """Calculates the (x, y) intersection pixel of two lines."""
        # If both lines are vertical, they are parallel and never intersect
        if line1.get("vertical") and line2.get("vertical"):
            return None
            
        # Case 1: Line 1 is vertical
        if line1.get("vertical"):
            x = line1["x_intercept"]
            y = line2["slope"] * x + line2["intercept"]
            return [int(x), int(y)]
            
        # Case 2: Line 2 is vertical
        if line2.get("vertical"):
            x = line2["x_intercept"]
            y = line1["slope"] * x + line1["intercept"]
            return [int(x), int(y)]
            
        # Case 3: Standard intersecting lines
        m1, b1 = line1["slope"], line1["intercept"]
        m2, b2 = line2["slope"], line2["intercept"]
        
        # If slopes are the same, they are parallel
        if m1 == m2: return None 
        
        # Algebra: m1*x + b1 = m2*x + b2  -->  x = (b2 - b1) / (m1 - m2)
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        
        return [int(x), int(y)]
    
    def supercharged_line_detection(self, frame):

        height, width = frame.shape[:2]
        cutoff_y = int(height * 0.40) # percentage (0.40 = 40%)

        # print(height, width)
       
        # Black out the crowd (Top 40% of the screen)
        roi_frame = frame.copy() 
        roi_frame[0:cutoff_y, :] = 0

        # hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv", hsv)
        
        # # # Find the floor
        # # floor_mask = cv2.inRange(hsv, self.floor_lower, self.floor_upper)
        # # cv2.imshow("floor_mask", floor_mask)

        # # Dilate the floor mask 
        # # Dilation expands the floor mask so it bleeds over and covers the white lines.
        # dilate_kernel = np.ones((25, 25), np.uint8)
        # floor_mask = cv2.dilate(hsv, dilate_kernel, iterations=1)
        # cv2.imshow("floor_mask_dilated", floor_mask)

        # # Convert to Grayscale (CLAHE works best on 1-channel images)
        # gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)

        b, g, r = cv2.split(roi_frame)
        gray = b
        # cv2.imshow("gray", gray)

        # Apply CLAHE to force the lines to pop out of shadows
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        # cv2.imshow("enhanced_gray", enhanced_gray)

        # Thresholding (Instead of HSV, we can use Adaptive Thresholding on the enhanced gray)
        mask = cv2.adaptiveThreshold(
            enhanced_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, -5
        )
        # cv2.imshow("adaptiveThreshold", mask)

        # MORPHOLOGICAL CLOSING
        # Create a 5x5 structural element (a small block of pixels)
        kernel = np.ones((5, 5), np.uint8)

        # Close the gaps caused by players walking on the lines
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("closed_mask", closed_mask)

        
        # Erode noise (removes tiny white dots like reflections on shoes)
        clean_mask = cv2.erode(closed_mask, np.ones((3, 3), np.uint8), iterations=1)
        # cv2.imshow("clean_mask", clean_mask)

        # # Apply the Floor Stencil: Only keep white lines that are ON the floor!
        # clean_floor_lines = cv2.bitwise_and(clean_mask, clean_mask, mask=floor_mask)
        # cv2.imshow("clean_floor_lines", clean_floor_lines)

        blurred_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)
        # cv2.imshow("closed_mask", blurred_mask)

        # Dynamic Canny Thresholding based on image median
        v = np.median(blurred_mask)
        lower_thresh = int(max(0, (1.0 - 0.33) * v))
        upper_thresh = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred_mask, lower_thresh, upper_thresh)
        # cv2.imshow("edges", edges)
        
        raw_lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50, 
            minLineLength=80,
            maxLineGap=25
        )

        # lsd = cv2.createLineSegmentDetector(0)
        # # Detect the lines
        # raw_lines, width, prec, nfa = lsd.detect(final_mask)
        # if raw_lines is None:
        #     return None

        formatted_lines = raw_lines

        # master_h, master_v = self.merge_overlapping_lines(formatted_lines, distance_threshold=15)
        # self.debug_draw_master_lines(debug_frame, master_h, master_v)

        if formatted_lines is None:
            return None

        return formatted_lines

    def debug_draw_hough_detection(self, frame, formatted_lines):
        debug_frame = frame.copy()
        for line in formatted_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if self.is_show_debug:
            cv2.imshow("debug_frame", debug_frame)
            cv2.waitKey(1)

        return debug_frame

    def debug_draw_master_lines(self, frame, master_h, master_v):
        """
        Converts mathematical slopes and intercepts back into pixel endpoints 
        and draws them across the entire frame for debugging.
        """
        # Create a copy so we don't permanently draw on the original frame
        debug_frame = frame.copy()
        height, width = debug_frame.shape[:2]

        # Draw Master Horizontal Lines (Equation: y = mx + b)
        # We calculate the Y position at the far left (X=0) and far right (X=width)
        for slope, intercept in master_h:
            y1 = int(intercept)                       # Y at X=0
            y2 = int(slope * width + intercept)       # Y at X=width
            
            # Draw in Blue (BGR format: 255, 0, 0), thickness 3
            cv2.line(debug_frame, (0, y1), (width, y2), (255, 0, 0), 3)

        # Draw Master Vertical Lines (Equation: x = my + b)
        # We calculate the X position at the very top (Y=0) and very bottom (Y=height)
        for slope, intercept in master_v:
            x1 = int(intercept)                       # X at Y=0
            x2 = int(slope * height + intercept)      # X at Y=height
            
            # Draw in Red (BGR format: 0, 0, 255), thickness 3
            cv2.line(debug_frame, (x1, 0), (x2, height), (0, 0, 255), 3)

        if self.is_show_debug:
            cv2.imshow("Debug: Merged Master Lines", debug_frame)
            cv2.waitKey(1)
        
        return debug_frame

    def debug_draw_fit_lines(self, frame, final_lines):
        debug_frame = frame.copy()
        height, width = debug_frame.shape[:2]

        for name, line in final_lines.items():
            if line.get("vertical"):
                x = int(line["x_intercept"])
                cv2.line(debug_frame, (x, 0), (x, height), (0, 0, 255), 3)
            else:
                slope = line["slope"]
                intercept = line["intercept"]
                if abs(slope) > 1.0:
                    # Calculate X at the top (Y=0) and bottom (Y=height)
                    y1 = 0
                    x1 = int((y1 - intercept) / slope)
                    y2 = height
                    x2 = int((y2 - intercept) / slope)
                    cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                else:
                    # Calculate Y at the far left (X=0) and far right (X=width)
                    x1 = 0
                    y1 = int(slope * x1 + intercept)
                    x2 = width
                    y2 = int(slope * x2 + intercept)
                    cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        if self.is_show_debug:
            cv2.imshow("Debug: Fitted Lines", debug_frame)
            cv2.waitKey(1)
        
        return debug_frame

    def debug_draw_corners(self, frame, corners):
        """
        Draws intersection corners
        """
        debug_frame = frame.copy()
        
        if corners is not None:
            # Loop through the 4 corners
            for i, pt in enumerate(corners):
                # Extract the raw x and y integers from the nested array
                x, y = int(pt[0][0]), int(pt[0][1])
                
                # Draw a circle
                cv2.circle(debug_frame, (x, y), 8, (255, 255, 0), -1)
                
                # Draw the index number
                cv2.putText(
                    debug_frame, str(i), (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
                )
                
        if self.is_show_debug:
            cv2.imshow("Debug: Intersections", debug_frame)
            cv2.waitKey(1)
        
        return debug_frame

    def calculate_intersections(self, horizontal_lines, vertical_lines):
        """
        Finds the (x, y) intersection of every horizontal line with every vertical line.
        """
        intersections = []

        for h_slope, h_y_intercept in horizontal_lines:
            for v_slope, v_x_intercept in vertical_lines:
                
                # Algebra: Solving the system of equations
                # Line 1 (Horizontal): y = (h_slope * x) + h_y_intercept
                # Line 2 (Vertical):   x = (v_slope * y) + v_x_intercept
                
                # Substitution to find Y:
                denom = 1 - (h_slope * v_slope)
                if denom == 0: 
                    continue # Lines are perfectly parallel, no intersection
                    
                y = (h_slope * v_x_intercept + h_y_intercept) / denom
                
                # Plug Y back in to find X:
                x = (v_slope * y) + v_x_intercept
                
                intersections.append([int(x), int(y)])

        return intersections
    
    def merge_overlapping_lines(self, lines, distance_threshold=40, angle_threshold=10):
        horizontal_lines = [] 
        vertical_lines = []   

        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate the length of the segment to use as a weight
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length == 0: 
                continue # Skip invalid zero-length lines

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)
            if angle > 90:
                angle = 180 - angle

            # HORIZONTAL LINES
            if angle < 45:
                if x2 - x1 == 0: continue 
                slope = (y2 - y1) / (x2 - x1)
                y_intercept = y1 - (slope * x1)
                
                matched = False
                for h_line in horizontal_lines:
                    # Use length-weighted averages for the existing cluster tracking
                    avg_intercept = np.average(h_line['intercepts'], weights=h_line['lengths'])
                    avg_slope = np.average(h_line['slopes'], weights=h_line['lengths'])
                    
                    # Check how different the angles are (prevents merging unaligned lines)
                    slope_diff_angle = np.degrees(np.abs(np.arctan(slope) - np.arctan(avg_slope)))

                    if abs(avg_intercept - y_intercept) < distance_threshold and slope_diff_angle < angle_threshold:
                        h_line['slopes'].append(slope)
                        h_line['intercepts'].append(y_intercept)
                        h_line['lengths'].append(length)
                        matched = True
                        break
                
                if not matched:
                    horizontal_lines.append({'slopes': [slope], 'intercepts': [y_intercept], 'lengths': [length]})

            # VERTICAL LINES
            else:
                if y2 - y1 == 0: continue 
                slope = (x2 - x1) / (y2 - y1) 
                x_intercept = x1 - (slope * y1)
                
                matched = False
                for v_line in vertical_lines:
                    avg_intercept = np.average(v_line['intercepts'], weights=v_line['lengths'])
                    avg_slope = np.average(v_line['slopes'], weights=v_line['lengths'])
                    slope_diff_angle = np.degrees(np.abs(np.arctan(slope) - np.arctan(avg_slope)))

                    if abs(avg_intercept - x_intercept) < distance_threshold and slope_diff_angle < angle_threshold:
                        v_line['slopes'].append(slope)
                        v_line['intercepts'].append(x_intercept)
                        v_line['lengths'].append(length)
                        matched = True
                        break
                        
                if not matched:
                    vertical_lines.append({'slopes': [slope], 'intercepts': [x_intercept], 'lengths': [length]})

        # Compute final master lines using length-weighted averages
        master_horizontal = []
        for h in horizontal_lines:
            w_slope = np.average(h['slopes'], weights=h['lengths'])
            w_intercept = np.average(h['intercepts'], weights=h['lengths'])
            master_horizontal.append((w_slope, w_intercept))

        master_vertical = []
        for v in vertical_lines:
            w_slope = np.average(v['slopes'], weights=v['lengths'])
            w_intercept = np.average(v['intercepts'], weights=v['lengths'])
            master_vertical.append((w_slope, w_intercept))

        return master_horizontal, master_vertical
    
    def filter_by_angle(self, raw_lines, horiz_tolerance=15, vert_tolerance=15):
        """
        Takes raw Hough lines and deletes any lines that are not strictly horizontal or vertical.
        horiz_tolerance: Allowable variance from 0 degrees (e.g., 15 means 0 to 15 degrees).
        vert_tolerance: Allowable variance from 90 degrees (e.g., 15 means 75 to 90 degrees).
        """
        if raw_lines is None:
            return None
            
        clean_lines = []
        
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate the angle using arctangent (returns radians, so we convert to degrees)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize the angle so it always sits between 0 and 180 degrees
            angle = abs(angle)
            if angle > 90:
                angle = 180 - angle
                
            # Check our buckets
            is_horizontal = angle <= horiz_tolerance
            is_vertical = (90 - angle) <= vert_tolerance
            
            # Only keep the line if it makes geometric sense
            if is_horizontal or is_vertical:
                clean_lines.append(line)
                
        return clean_lines

    def match_lines_to_ghost_court(self, raw_lines, current_H, distance_thresh=10, angle_thresh=10):
        """Project Ghost Lines and calculate their dynamic angles"""
        if raw_lines is None: return None

        _, H_inv = cv2.invert(current_H)
        
        ghost_lines = {}
        for name, line_pts in self.blueprint.items():
            pts = np.array(line_pts, dtype=np.float32).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(pts, H_inv)
            
            px1, py1 = projected[0][0]
            px2, py2 = projected[1][0]
            
            ghost_lines[name] = {
                "p1": (px1, py1),
                "p2": (px2, py2),
                "angle": self._get_line_angle(px1, py1, px2, py2) # Dynamic expected angle!
            }

        matched_results = {name: [] for name in self.blueprint.keys()}
        
        # Filter raw lines against Ghost Lines
        for line in raw_lines:
            hx1, hy1, hx2, hy2 = line[0]
            h_angle = self._get_line_angle(hx1, hy1, hx2, hy2)
                
            for name, ghost in ghost_lines.items():
                # Calculate angle difference
                angle_diff = self._calculate_angle_diff(h_angle, ghost["angle"])
                
                # Check Angle
                if angle_diff < angle_thresh:
                    # Check Distance 
                    dist1 = self._point_to_line_dist(hx1, hy1, ghost["p1"], ghost["p2"])
                    dist2 = self._point_to_line_dist(hx2, hy2, ghost["p1"], ghost["p2"])
                    
                    if dist1 < distance_thresh and dist2 < distance_thresh:
                        matched_results[name].append(line[0])
                        break
                        
        return matched_results

    def fit_lines(self, categorized_lines):
        """
        Takes the sorted buckets of raw segments and fits them into one line using cv2.fitLine.
        """
        final_court_lines = {}
        
        # print("--- name, len(line_segments)")
        for name, line_segments in categorized_lines.items():
            # print(name, len(line_segments))
            if len(line_segments) == 0:
                continue # We didn't find any pieces of this line on this frame
                
            # Extract all the (x,y) points from the fragments in this bucket
            all_points = []
            for segment in line_segments:
                x1, y1, x2, y2 = segment
                all_points.append([x1, y1])
                all_points.append([x2, y2])
                
            points_array = np.array(all_points, dtype=np.int32)
            
            # Fit a single, perfect infinite line through all points
            # Returns a normalized vector (vx, vy) and a point on the line (x0, y0)
            [vx, vy, x0, y0] = cv2.fitLine(points_array, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Convert vector format to slope and intercept (for intersection math)
            if vx[0] != 0:
                slope = vy[0] / vx[0]
                intercept = y0[0] - (slope * x0[0])
                final_court_lines[name] = {"slope": slope, "intercept": intercept, "vertical": False}
            else:
                # Handle perfectly vertical lines to prevent divide-by-zero
                final_court_lines[name] = {"x_intercept": x0[0], "vertical": True}
                
        return final_court_lines

    def extract_corners(self, final_court_lines):
        """
        Looks at the available lines, determines the visible zone, 
        and returns both the camera points and the 2D canvas points
        """
        keys = final_court_lines.keys()

        # Must have both sidelines to form a 4-corner polygon
        if "top_sideline" not in keys or "bottom_sideline" not in keys:
            return None, None
            
        top = final_court_lines["top_sideline"]
        bot = final_court_lines["bottom_sideline"]

        # Figure out which verticals detected to define the zone
        verticals_found = []
        dst_pts_list = []

        # Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        # Check for Full Court
        if "left_baseline" in keys and "right_baseline" in keys:
            verticals_found = [final_court_lines["left_baseline"], final_court_lines["right_baseline"]]
            # 2D Blueprint Coordinates (Scale: 100px = 1m)
            dst_pts_list = [[0, 0], [1800, 0], [0, 900], [1800, 900]]

        # Left back + left attack + right attack areas
        elif "left_baseline" in keys and "right_attackline" in keys:
            verticals_found = [final_court_lines["left_baseline"], final_court_lines["right_attackline"]]
            dst_pts_list = [[0, 0], [1200, 0], [0, 900], [1200, 900]]

        # Left attack + right attack + right back areas
        elif "left_attackline" in keys and "right_baseline" in keys:
            verticals_found = [final_court_lines["left_attackline"], final_court_lines["right_baseline"]]
            dst_pts_list = [[600, 0], [1800, 0], [600, 900], [1800, 900]]

        # # Left back + left attack areas
        # elif "left_baseline" in keys and "center_line" in keys:
        #     verticals_found = [final_court_lines["left_baseline"], final_court_lines["center_line"]]
        #     dst_pts_list = [[0, 0], [900, 0], [0, 900], [900, 900]]

        # # Right attack + right back areas
        # elif "center_line" in keys and "right_baseline" in keys:
        #     verticals_found = [final_court_lines["center_line"], final_court_lines["right_baseline"]]
        #     dst_pts_list = [[900, 0], [1800, 0], [900, 900], [1800, 900]]

        # left attack + right attack areas
        elif "left_attackline" in keys and "right_attackline" in keys:
            verticals_found = [final_court_lines["left_attackline"], final_court_lines["right_attackline"]]
            dst_pts_list = [[600, 0], [1200, 0], [600, 900], [1200, 900]]

        # Left back area
        elif "left_baseline" in keys and "left_attackline" in keys:
            verticals_found = [final_court_lines["left_baseline"], final_court_lines["left_attackline"]]
            dst_pts_list = [[0, 0], [600, 0], [0, 900], [600, 900]]

        # Right back area
        elif "right_attackline" in keys and "right_baseline" in keys:
            verticals_found = [final_court_lines["right_attackline"], final_court_lines["right_baseline"]]
            dst_pts_list = [[1200, 0], [1800, 0], [1200, 900], [1800, 900]]

        # # Left attack area
        # elif "left_attackline" in keys and "center_line" in keys:
        #     verticals_found = [final_court_lines["left_attackline"], final_court_lines["center_line"]]
        #     dst_pts_list = [[600, 0], [900, 0], [600, 900], [900, 900]]

        # # Right attack area
        # elif "center_line" in keys and "right_attackline" in keys:
        #     verticals_found = [final_court_lines["center_line"], final_court_lines["right_attackline"]]
        #     dst_pts_list = [[900, 0], [1200, 0], [900, 900], [1200, 900]]
        else:
            return None, None

        # Calculate the actual pixel intersections using found verticals
        left_v = verticals_found[0]
        right_v = verticals_found[1]

        tl = self._calculate_intersection(top, left_v)
        tr = self._calculate_intersection(top, right_v)
        bl = self._calculate_intersection(bot, left_v)
        br = self._calculate_intersection(bot, right_v)

        if None in [tl, tr, bl, br]:
            return None, None

        # Format the outputs for OpenCV
        # Shape must be (4, 1, 2) and float32
        src_pts = np.array([tl, tr, bl, br], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(dst_pts_list, dtype=np.float32).reshape(-1, 1, 2)

        return src_pts, dst_pts

    def detect(self, frame, current_H):
        raw_lines = self.supercharged_line_detection(frame)
        debug_frame = self.debug_draw_hough_detection(frame, raw_lines)
        if callable(self.on_debug):
            self.on_debug("Hough Raw", debug_frame)

        if raw_lines is not None:
            # Ghost Matcher sorts them into buckets
            categorized_buckets = self.match_lines_to_ghost_court(raw_lines, current_H)
            
            # Fit the grouped detected lines 
            final_lines = self.fit_lines(categorized_buckets)
            debug_frame = self.debug_draw_fit_lines(frame, final_lines)
            if callable(self.on_debug):
                self.on_debug("Fitted Lines", debug_frame)
            
            # Extract new detected corners
            detected_corner, dst_pts = self.extract_corners(final_lines)
            debug_frame = self.debug_draw_corners(frame, detected_corner)
            if callable(self.on_debug):
                self.on_debug("Intersections", debug_frame)
            
            # Feeding to the next cycle
            if detected_corner is not None:
                try:
                    src_pts32 = np.array(detected_corner, dtype=np.float32)
                    dst_pts32 = np.array(dst_pts, dtype=np.float32)
                    # Update the Homography Matrix back to detected edges
                    # current_H, _ = cv2.findHomography(detected_corner, dst_pts)
                    new_H = cv2.getPerspectiveTransform(src_pts32, dst_pts32)

                    return new_H, src_pts32, dst_pts32
                except cv2.error as e:
                    print(f"OpenCV Error caught: {e}")

        return None, None, None
        
