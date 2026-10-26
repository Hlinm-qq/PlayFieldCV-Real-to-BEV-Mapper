import cv2
import numpy as np
    
class Visualizer:
    def __init__(self, math_len=1800, math_width=900, math_pad=300, orientation="horizontal"):
        """
        Args:
            math_len: Court baseline-to-baseline length
            math_width: Court sideline-to-sideline width
            math_pad: Out-of-bounds free zone on all sides
                        Increase this to zoom out and see more out-of-bounds space;
                        decrease to zoom in tightly on the court.
            orientation: 'horizontal' or 'vertical'.
        """
        self.math_len = math_len
        self.math_width = math_width
        self.math_pad = math_pad
        self.orientation = orientation

        # Total viewable range including out-of-bounds areas
        self.total_len = self.math_len + (2 * self.math_pad)
        self.total_width = self.math_width + (2 * self.math_pad)

        # OpenCV Sub-pixel anti-aliasing (16 sub-steps)
        self.SHIFT = 4
        self.SCALE = 1 << self.SHIFT

    def render(self, target_w: int, target_h: int, players: list[dict]) -> np.ndarray:
        # Canvas color
        canvas = np.full((target_h, target_w, 3), (80, 150, 220), dtype=np.uint8)
        court_color = (220, 150, 80)  # BGR
        line_color = (255, 255, 255)

        # Fit total viewable area (court + padding) into the target UI cell
        if self.orientation == "horizontal":
            total_aspect = self.total_len / self.total_width
        else:
            total_aspect = self.total_width / self.total_len

        if (target_w / target_h) > total_aspect:
            view_h = target_h
            view_w = int(view_h * total_aspect)
        else:
            view_w = target_w
            view_h = int(view_w / total_aspect)

        # Offsets to center the entire viewable box inside the cell
        ox = (target_w - view_w) // 2
        oy = (target_h - view_h) // 2

        # Math -> Canvas Sub-Pixel Mapping
        def tf(math_x, math_y):
            # Normalize with respect to the total bounding area (including padding)
            nx = (math_x + self.math_pad) / self.total_len
            ny = (math_y + self.math_pad) / self.total_width

            if self.orientation == "vertical":
                # Rotated 90 degrees: width along X, length along Y
                px = ox + (ny * view_w)
                py = oy + ((1.0 - nx) * view_h)
            else:
                px = ox + (nx * view_w)
                py = oy + (ny * view_h)

            return (int(round(px * self.SCALE)), int(round(py * self.SCALE)))

        # Draw Court Floor (Only the playing field inside lines, not the padding)
        c_top_left = tf(0, 0)
        c_bot_right = tf(self.math_len, self.math_width)
        
        # Recover integer pixel corners for floor fill
        p1 = (c_top_left[0] >> self.SHIFT, c_top_left[1] >> self.SHIFT)
        p2 = (c_bot_right[0] >> self.SHIFT, c_bot_right[1] >> self.SHIFT)
        
        # Draw colored surface
        cv2.rectangle(canvas, p1, p2, court_color, -1)

        # Draw Perimeter Lines
        corners = [
            tf(0, 0),
            tf(self.math_len, 0),
            tf(self.math_len, self.math_width),
            tf(0, self.math_width)
        ]
        for i in range(4):
            cv2.line(canvas, corners[i], corners[(i + 1) % 4], line_color, 3, cv2.LINE_AA, shift=self.SHIFT)

        # Draw Net & Attack Lines
        net_x = self.math_len / 2
        attack_offset = self.math_len / 6

        cv2.line(canvas, tf(net_x, 0), tf(net_x, self.math_width), (240, 240, 240), 5, cv2.LINE_AA, shift=self.SHIFT)
        cv2.line(canvas, tf(net_x - attack_offset, 0), tf(net_x - attack_offset, self.math_width), line_color, 2, cv2.LINE_AA, shift=self.SHIFT)
        cv2.line(canvas, tf(net_x + attack_offset, 0), tf(net_x + attack_offset, self.math_width), line_color, 2, cv2.LINE_AA, shift=self.SHIFT)

        # Draw Agendas
        text1 = "LEFT"
        text2 = "RIGHT"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, view_w * 0.0006)
        font_thickness = 2
        text_size2 = cv2.getTextSize(text2, font, font_scale, font_thickness)[0]
        
        # Place labels
        l_pt = tf(0, 0)
        r_pt = tf(self.math_len, 0)
        r_pt_v = tf(self.math_len, self.math_width)
        lx, ly = l_pt[0] >> self.SHIFT, l_pt[1] >> self.SHIFT
        rx, ry = r_pt[0] >> self.SHIFT, r_pt[1] >> self.SHIFT
        rx_v, ry_v = r_pt_v[0] >> self.SHIFT, r_pt_v[1] >> self.SHIFT

        if self.orientation == "vertical":
            cv2.putText(canvas, text1, (lx, ly + 15), font, font_scale, (255, 0, 0), font_thickness)
            cv2.putText(canvas, text2, (rx, ry - 5), font, font_scale, (0, 0, 255), font_thickness)
        else:
            cv2.putText(canvas, text1, (lx, ly - 5), font, font_scale, (255, 0, 0), font_thickness)
            cv2.putText(canvas, text2, (rx - text_size2[0], ry - 5), font, font_scale, (0, 0, 255), font_thickness)

        # Draw Players
        dot_radius = max(3, int(min(target_w, target_h) * 0.005))
        sub_radius = dot_radius * self.SCALE

        for p in players:
            center = tf(p["math_x"], p["math_y"])
            px = center[0] >> self.SHIFT
            py = center[1] >> self.SHIFT

            # Draw if player is anywhere within visible target cell bounds
            if 0 <= px < target_w and 0 <= py < target_h:
                cv2.circle(canvas, center, sub_radius + (2 * self.SCALE), (10, 10, 10), -1, cv2.LINE_AA, shift=self.SHIFT)
                cv2.circle(canvas, center, sub_radius, p["color"], -1, cv2.LINE_AA, shift=self.SHIFT)

                if "id" in p :
                    if "team" in p["team_id"]:
                        cv2.putText(canvas, str(p["id"]), (px - 6, py - 12),
                                    font, font_scale * 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

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






