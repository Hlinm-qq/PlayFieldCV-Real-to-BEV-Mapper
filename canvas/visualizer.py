import cv2
import numpy as np
    
class Visualizer:
    def __init__(self, width, height, padding=0):
        self.width = width
        self.height = height
        self.padding = padding
        self.content_size = [width, height]

        self.padding = self.padding if self.padding >= 0 else 0
        padding_space = self.padding*2
        if padding_space < self.content_size[0] and padding_space < self.content_size[1]:
            self.content_size[0] = self.content_size[0] - padding_space
            self.content_size[1] = self.content_size[1] - padding_space

    def draw_empty_court(self):
        """Draws a fresh 2D volleyball court canvas."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        padding = self.padding if self.padding >= 0 else 0

        range_to = (padding + self.content_size[0], padding + self.content_size[1])
        canvas[padding:range_to[1], padding:range_to[0]] = (80, 150, 220) # Orange/wood color
        
        # Calculate dynamic positions
        center_y = padding + int(self.content_size[1] / 2)
        
        # The attack line is exactly 1/6th of the court's total length away from the net
        attack_line_offset = padding + int(self.content_size[1] / 6) 
        
        top_attack_line = center_y - attack_line_offset
        bottom_attack_line = center_y + attack_line_offset
        
        line_color = (255, 255, 255)
        
        # Draw Lines
        cv2.rectangle(canvas, (padding, padding), (range_to[0], range_to[1]), line_color, 2)            # Outer bounds
        cv2.line(canvas, (padding, center_y), (range_to[0], center_y), line_color, 4)             # Center Net
        cv2.line(canvas, (padding, top_attack_line), (range_to[0], top_attack_line), line_color, 2)       # Top 3m Line
        cv2.line(canvas, (padding, bottom_attack_line), (range_to[0], bottom_attack_line), line_color, 2) # Bottom 3m Line
        return canvas

    def draw_player_on_map(self, canvas, map_x, map_y, player_id):
        """Draws a dot and ID on the 2D map."""
        # Only draw if the point is actually inside the court boundaries
        if 0 <= map_x <= self.width and 0 <= map_y <= self.height:
            cv2.circle(canvas, (map_x, map_y), 8, (0, 255, 0), -1) # Green dot
            cv2.putText(canvas, str(player_id), (map_x - 10, map_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas
    
    def get_content_size(self):
        return self.content_size


class FieldProjector:
    def __init__(self, src_points, canvas_width, canvas_height, padding=0):
        self.width = canvas_width
        self.height = canvas_height
        self.padding = padding
        
        # Define destination points
        # top-left
        # top-right
        # bottom-right
        # bottom-left
        dst_points = np.float32([
            [self.padding, self.padding], 
            [self.width - self.padding, self.padding], 
            [self.width - self.padding, self.height - self.padding], 
            [self.padding, self.height - self.padding]
        ])
        
        # Generate the magic matrix once
        self.matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def get_map_coordinate(self, x, y):
        """Converts a real-world foot coordinate to a 2D map coordinate."""
        # OpenCV requires a very specific array shape for this function
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        mapped_point = cv2.perspectiveTransform(point, self.matrix)
        
        # Extract the new X and Y
        map_x = int(mapped_point[0][0][0])
        map_y = int(mapped_point[0][0][1])
        return map_x, map_y







