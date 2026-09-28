import cv2
import numpy as np
    
class Visualizer:
    def __init__(self, court_length=1800, court_width=900, padding=50, orientation="horizontal"):
        """
        court_length: The long side of the court (X-axis in math)
        court_width: The short side of the court (Y-axis in math)
        orientation: "horizontal" or "vertical"
        """
        self.orientation = orientation
        self.padding = padding
        self.math_len = court_length
        self.math_wid = court_width

        # Screen dimensions flip based on orientation
        if self.orientation == "vertical":
            self.screen_w = court_width + (padding * 2)
            self.screen_h = court_length + (padding * 2)
        else:
            self.screen_w = court_length + (padding * 2)
            self.screen_h = court_width + (padding * 2)

    def draw_empty_court(self):
        """Draws a fresh 2D volleyball court canvas."""
        canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                
        # Background Color
        canvas[:] = (80, 150, 220) 
        # Line Color
        line_color = (255, 255, 255)

        p = self.padding
        w = self.screen_w - p
        h = self.screen_h - p

        # Draw Outer Boundary
        cv2.rectangle(canvas, (p, p), (w, h), line_color, 4)
   
        if self.orientation == "vertical": # "vertical"
            # Net is a horizontal line in the middle
            net_y = p + (self.math_len // 2)
            cv2.line(canvas, (p, net_y), (w, net_y), line_color, 6)
            # Attack lines
            attack_offset = self.math_len // 6
            cv2.line(canvas, (p, net_y - attack_offset), (w, net_y - attack_offset), line_color, 2)
            cv2.line(canvas, (p, net_y + attack_offset), (w, net_y + attack_offset), line_color, 2)
        else: # horizontal
            # Net is a vertical line in the middle
            net_x = p + (self.math_len // 2)
            # print("net_x")
            # print(net_x)
            cv2.line(canvas, (net_x, p), (net_x, h), line_color, 6)
            # Attack lines (3 meters = 1/6 of length)
            attack_offset = self.math_len // 6
            cv2.line(canvas, (net_x - attack_offset, p), (net_x - attack_offset, h), line_color, 2)
            cv2.line(canvas, (net_x + attack_offset, p), (net_x + attack_offset, h), line_color, 2)

        return canvas

    def draw_player_on_map(self, canvas, math_x, math_y, player_id, color: tuple = (0, 255, 0)):
        """
        Draws a dot and ID on the 2D map, math_x and math_y is horizontal based.
        shifts them into the pixel window, and draws them if they fit on screen.
        """
        # Convert Math Space to Pixel Space (Handling the orientation flip)
        if self.orientation == "vertical":
            draw_x = self.padding + math_y
            draw_y = self.padding + (self.math_len - math_x)
            
        else: # Rotate 90 degrees clockwise
            draw_x = self.padding + math_x
            draw_y = self.padding + math_y

        # Convert to integers for drawing pixels
        draw_x = int(draw_x)
        draw_y = int(draw_y)

        # Limit Check
        if 0 <= draw_x <= self.screen_w and 0 <= draw_y <= self.screen_h:
            # Draw the player (as dot)
            cv2.circle(canvas, (draw_x, draw_y), 8, color, -1)

            text = str(player_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            cv2.putText(canvas, text, 
                        (draw_x - (text_size[0] // 2), draw_y - 15), 
                        font, 0.6, (255, 255, 255), 2)
        else:
            pass 
            
        return canvas
    
    def get_content_size(self):
        return self.content_size

class FieldProjector:
    def __init__(self, src_points, canvas_width, canvas_height, dst_points=None, padding=0, is_dynamic=False):
        self.matrix = None

        if not is_dynamic:
            self.width = canvas_width
            self.height = canvas_height
            self.padding = padding
            
            # Define destination points
            # top-left
            # top-right
            # bottom-right
            # bottom-left
            if dst_points is None:
                dst_points = np.float32([
                    [self.padding, self.padding], 
                    [self.width - self.padding, self.padding], 
                    [self.width - self.padding, self.height - self.padding], 
                    [self.padding, self.height - self.padding]
                ])

            # Generate the matrix once
            self.matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        else:
            pass

    def get_map_coordinate(self, x, y, current_H=None):
        """Converts a real-world foot coordinate to a 2D map coordinate."""
        curr_matrix = None

        if current_H is not None:
            curr_matrix = current_H    
        elif self.matrix is not None:
            curr_matrix = self.matrix

        if curr_matrix is None:
            return None, None
            
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        
        # Apply the current frame's perspective matrix
        mapped_point = cv2.perspectiveTransform(point, curr_matrix)
        
        map_x = int(mapped_point[0][0][0])
        map_y = int(mapped_point[0][0][1])
        
        return map_x, map_y






