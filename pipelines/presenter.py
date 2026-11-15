import time
import random, numpy as np, torch, cv2

from config.volleyball import *
from utils.utils import PipelineConfig
from data_manager.output_manager import *
from canvas.visualizer import *
from pipelines.volleyball_pipeline import TeamClassifier

class Presenter:
    def __init__(self, 
                 target_fps = 60, 
                 visualizer: Visualizer = None, 
                 dashboard: AsymmetricDashboard = None, 
                 debug_manager: PipelineDebugger = None, 
                 output_manager: OutputManager = None, 
                 team_classifier: TeamClassifier = None, 
                 config: PipelineConfig = None
                ):
        # Target configs
        self.target_fps = target_fps
        self.target_time = (1.0 / target_fps)

        # Handler
        self.visualizer = visualizer
        self.dashboard = dashboard
        self.debug_manager = debug_manager
        self.output_manager = output_manager
        self.team_classifier = team_classifier
        self.config = config

        # Mandatory Checking
        self.is_initialized = False

    def initialize(self):
        if None in (self.visualizer, self.dashboard, self.debug_manager, self.output_manager, self.team_classifier, self.config):
            return 
        self.is_initialized = True

    def present(self, frame, result, view_idx, loop_start) -> bool:
        # Draw player boxes, ground anchors, and build BEV payload
        map_players_data = []
        for p in result.players:
            cv2.rectangle(frame, (int(p.bbox.x1), int(p.bbox.y1)),
                            (int(p.bbox.x2), int(p.bbox.y2)), (0, 0, 255), 2)
            cv2.circle(frame, (int(p.src_coords.x), int(p.src_coords.y)), 5, (255, 0, 0), -1)
    
            if p.track_id != -1:
                cv2.putText(frame, f"ID: {p.track_id}", (int(p.bbox.x1), int(p.bbox.y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
            if p.team_id == "":
                p.team_id = self.team_classifier.get_locked_team(p.track_id)
    
            team_key = p.team_id.lower()
            color = COLOR_CONFIG.get(team_key, (0, 0, 0))
    
            map_players_data.append({
                "math_x": p.map_coords.x,
                "math_y": p.map_coords.y,
                "id": p.track_id,
                "color": color,
                "team_id": team_key
            })
    
        # Throttle to target FPS BEFORE computing final displayed FPS
        process_time = time.perf_counter() - loop_start
        if self.target_time > 0 and process_time < self.target_time:
            time.sleep(self.target_time - process_time)
    
        total_loop_time = time.perf_counter() - loop_start
        total_loop_ms = total_loop_time * 1000.0
        actual_fps = 1.0 / total_loop_time if total_loop_time > 0 else 0.0
    
        # Draw telemetry on frame
        cv2.putText(frame, f"View Frame: {view_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Target FPS: {self.target_fps if self.target_fps > 0 else 'MAX'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Actual FPS: {int(actual_fps)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
        print(f"View Frame {view_idx} | Actual FPS: {actual_fps:.1f} ({total_loop_ms:.1f} ms)")
    
        # Render BEV canvas & compose dashboard
        map_bev = self.visualizer.render(target_w=self.dashboard.hero_w, target_h=self.dashboard.hero_h, players=map_players_data)
        self.dashboard.add_view("hero", map_bev, "Volleyball BEV")
        self.dashboard.add_view("aux", frame, "Raw Detection")
    
        combined_view = self.dashboard.render()
        self.debug_manager.show()
    
        # Write video frame
        if self.config.save_video and self.config.is_combine_result:
            self.output_manager.video_out(combined_view)
    
        # Display frame & handle single unified exit check
        if self.config.is_combine_result:
            key_pressed = self.output_manager.display_frame(combined_view, "Combined Output")
        else:
            key_pressed = self.output_manager.display_frame(frame, "Original Detection")
    
        if key_pressed == ord('q'):
            return True
    
        if self.config.sleep_second > 0:
            if not self.output_manager.get_is_notebook():
                if cv2.waitKey(int(1000 * self.config.sleep_second)) & 0xFF == ord('q'):
                    return True
            else:
                time.sleep(self.config.sleep_second)
    
        return False