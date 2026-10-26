import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time

class OutputManager:
    def __init__(self):
        self.display_cell = None
        self.is_notebook = -1

        self.display = None
        self.Image = None
        self.clear_output = None

        self.video_writer = None
        self.video_extension = 'mp4'

        if self.is_notebook == -1:
            env_checker = self.check_is_notebook(False)
            self.is_notebook = 1 if env_checker else 0
            if env_checker:
                # Keep them as class attributes
                from IPython.display import display, Image, clear_output
                self.display = display
                self.Image = Image
                self.clear_output = clear_output
                
                # Create a persistent handle for updating the same cell
                self.display_cell = self.display(display_id=True)

    def check_is_notebook(self, is_print=True):
        try:
            # Check if the IPython kernel is running
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                if is_print:
                    print("yes: ZMQInteractiveShell")
                return True
            else:
                # Terminal running IPython
                if is_print:
                    print("yes: IPython")
                return True
        except NameError:
            None
            return False

    def display_frame(self, frame, window_name="Window", wait_ms=1):
        """
        Dynamic Single frame display helper
        """
        if self.is_notebook == 1 and self.display is not None and self.Image is not None:
            # --- JUPYTER NOTEBOOK ---
            ret, encoded_image = cv2.imencode('.jpg', frame)
            if ret:
                # clear_output(wait=True)
                if self.display_cell is not None:
                    self.display_cell.update(self.Image(data=encoded_image.tobytes()))
                else:
                    self.display(self.Image(data=encoded_image.tobytes()))
                    print("cell is none")
                
            return -1 
            
        else:
            # --- STANDARD IDE / TERMINAL ---
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            # cv2.resizeWindow(window_name, 1280, 720)
            cv2.imshow(window_name, frame)
            
            safe_wait = max(1, int(wait_ms)) 
        return cv2.waitKey(safe_wait) & 0xFF
    
    def get_is_notebook(self):
        return self.is_notebook == 1
    
    def save_image(self, image, folder="output", filename=None, prefix="img"):
        """
        Saves an OpenCV image, handles directory creation and naming.
        """
        # Validation: avoid empty image
        if image is None:
            print("Save Error: Image is empty (None).")
            return False

        # Path Handling
        save_path = Path(folder)
        save_path.mkdir(parents=True, exist_ok=True)

        # Naming: If no name is given, use a timestamp
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}.jpg"

        full_path = save_path / filename
        success = cv2.imwrite(str(full_path), image)
        
        if success:
            # print(f"Successfully saved: {full_path}")
            None
        else:
            # print(f"Failed to save image to: {full_path}")
            None
            
        return success
    
    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()

    def init_video_writer(self, target_fps, target_w, target_h, folder="output", filename=None, codec='mp4v'):
        valid_codecs = {'XVID': 'avi', 
                      'mp4v': 'mp4', 
                      'MJPG': 'mp4', # 'mp4' or 'avi'
                      'avc1': 'mp4', 
                      'DIVX': 'avi', 
                      'X264': 'mp4'}
        
        # Normalize input and validate
        # codec = codec.lower()
        if codec not in valid_codecs:
            print(f"Codec {codec} not supported. Use: {list(valid_codecs.keys())}")
            return False
        
        if target_fps <= 0:
            target_fps = 30
        
        print(f"Video config:\nFPS: {target_fps}, \nWidth: {target_w}, \nHeight: {target_h}")

        # Path Handling
        self.video_extension = valid_codecs[codec]
        save_path = Path(folder)
        save_path.mkdir(parents=True, exist_ok=True)

        # Naming: If no name is given, use a timestamp
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.{self.video_extension}"
        full_path = save_path / filename

        # Create the FourCC object
        fourcc = cv2.VideoWriter_fourcc(*codec)

        # Format: (Path, FourCC, FPS, (Width, Height))
        self.video_writer = cv2.VideoWriter(
            full_path, 
            fourcc, 
            target_fps, 
            (int(target_w), int(target_h))
        )

    def video_out(self, frame):
        if self.video_writer is not None:
            self.video_writer.write(frame)

class PipelineDebugger:
    """Aggregates intermediate pipeline stages and arranges them into a 
    resolution-capped grid.
    """
    PRESETS = {
        "svga":  (800, 600),
        "720p":  (1280, 720),
        "wxga":  (1366, 768),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4k":    (3840, 2160),
    }

    def __init__(
        self,
        enabled=True,
        resolution="1080p",
        cols=3,
        rows=3,
        keep_aspect=True
    ):
        self.enabled = enabled
        self.cols = max(1, cols)
        self.rows = max(1, rows) if rows is not None else None
        self.keep_aspect = keep_aspect
        self.frames = {}

        # Resolve total canvas bounds
        if isinstance(resolution, str):
            res_key = resolution.lower()
            if res_key not in self.PRESETS:
                res_key = "1080p"
                # raise ValueError(f"Unknown preset '{resolution}'. Available: {list(self.PRESETS.keys())}")
            self.max_w, self.max_h = self.PRESETS[res_key]
        elif isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            self.max_w, self.max_h = int(resolution[0]), int(resolution[1])
        else:
            raise TypeError("resolution must be a preset string or a (width, height) tuple")

    def add(self, name: str, frame: np.ndarray, target_size: tuple = None, mode: str = "fit"):
        """Register an image with an optional custom size (w, h) and sizing mode.
        
        Args:
            name: Label displayed on the tile banner.
            frame: OpenCV image matrix (1-ch, 3-ch, or 4-ch).
            target_size: Optional (width, height). Clamped to cell ceiling if exceeded.
            mode: 'fit' (proportional letterbox), 'crop' (scale & cover),
                    or 'center_crop' (1:1 unscaled pixel crop).
        """
        if not self.enabled or frame is None:
            return
        self.frames[name] = {
            "img": frame.copy(),
            "target_size": target_size,
            "mode": mode.lower() 
        }

    def _prepare_tile(self, name: str, config: dict, cell_w: int, cell_h: int) -> np.ndarray:
        raw = config["img"]
        mode = config["mode"]
        req_size = config["target_size"]

        # Normalize types & bit depth
        if raw.dtype == bool:
            raw = (raw.astype(np.uint8)) * 255
        elif np.issubdtype(raw.dtype, np.floating):
            raw = np.clip(raw * 255 if raw.max() <= 1.0 else raw, 0, 255).astype(np.uint8)

        # Standardize to 3 channels (BGR)
        if len(raw.shape) == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif len(raw.shape) == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        # Resize
        h, w = raw.shape[:2]

        if req_size is not None:
            tw = min(int(req_size[0]), cell_w)
            th = min(int(req_size[1]), cell_h)
        else:
            tw, th = cell_w, cell_h
        
        if mode == "center_crop":
            # 1:1 true-pixel extraction around the center
            crop_w = min(w, tw)
            crop_h = min(h, th)
            x1 = max(0, (w - crop_w) // 2)
            y1 = max(0, (h - crop_h) // 2)
            processed = raw[y1:y1 + crop_h, x1:x1 + crop_w]

        elif mode == "crop":
            # Scale to completely fill (tw, th), then center crop the excess
            scale = max(tw / w, th / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            scaled = cv2.resize(raw, (nw, nh), interpolation=interp)

            # Center crop to target box
            cx, cy = nw // 2, nh // 2
            x1 = cx - (tw // 2)
            y1 = cy - (th // 2)
            processed = scaled[y1:y1 + th, x1:x1 + tw]

        else:  # "fit"
            # Scale down to fit inside (tw, th) by aspect ratio
            scale = min(tw / w, th / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            processed = cv2.resize(raw, (nw, nh), interpolation=interp)

        # Center processed content
        cell_canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        ph, pw = processed.shape[:2]
        offset_x = (cell_w - pw) // 2
        offset_y = (cell_h - ph) // 2
        cell_canvas[offset_y:offset_y + ph, offset_x:offset_x + pw] = processed

        # Stage Title Overlay
        cv2.rectangle(cell_canvas, (0, 0), (cell_w, 24), (20, 20, 20), -1)
        badge = f"{name}"
        cv2.putText(
            cell_canvas, badge, (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
        )
        return cell_canvas

    def render_grid(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((self.max_h, self.max_w, 3), dtype=np.uint8)

        active_rows = self.rows if self.rows else (len(self.frames) + self.cols - 1) // self.cols
        cell_w = self.max_w // self.cols
        cell_h = self.max_h // active_rows

        tiles = [
            self._prepare_tile(name, cfg, cell_w, cell_h)
            for name, cfg in self.frames.items()
        ]

        # Fill remaining grid cells
        total_slots = active_rows * self.cols
        if len(tiles) > total_slots:
            tiles = tiles[:total_slots]
        elif len(tiles) < total_slots:
            blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            tiles.extend([blank] * (total_slots - len(tiles)))

        # Stack into rows, then stack vertically
        row_strips = [
            np.hstack(tiles[r * self.cols : (r + 1) * self.cols])
            for r in range(active_rows)
        ]
        return np.vstack(row_strips)

    def show(self, window_name="Pipeline Debugger", wait_ms=1):
        if not self.enabled or not self.frames:
            return
        cv2.imshow(window_name, self.render_grid())
        return cv2.waitKey(wait_ms)

    def clear(self):
        self.frames.clear()

class AsymmetricDashboard:
    """Manages the output view layout"""
    PRESETS = {
        "480p":  (640, 480),
        "600p":  (800, 600),
        "720p":  (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
    }

    def __init__(self, resolution="1080p", hero_ratio=0.65, hud_height=40):
        if isinstance(resolution, str):
            res_key = resolution.lower()
            if res_key not in self.PRESETS:
                res_key = "720p"
                # raise ValueError(f"Unknown preset '{resolution}'. Available: {list(self.PRESETS.keys())}")
            self.total_w, self.total_h = self.PRESETS[res_key]
        elif isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            self.total_w, self.total_h = int(resolution[0]), int(resolution[1])
        else:
            raise TypeError("resolution must be a preset string or a (width, height) tuple")

        self.hud_height = hud_height
        self.grid_h = self.total_h - hud_height
        
        self.hero_w = int(self.total_w * hero_ratio)
        self.hero_h = self.grid_h
        self.aux_w = self.total_w - self.hero_w
        
        self.metrics = {}
        self.views = {}
        self.prev_tick = time.perf_counter()

    def set_metric(self, key: str, value):
        self.metrics[key] = value

    def add_view(self, name: str, frame: np.ndarray, label: str = "", is_mask: bool = False):
        if frame is not None:
            self.views[name] = {"img": frame, "label": label, "is_mask": is_mask}

    def _render_hud(self) -> np.ndarray:
        hud = np.full((self.hud_height, self.total_w, 3), 15, dtype=np.uint8)
        now = time.perf_counter()
        fps = 1.0 / max(1e-5, (now - self.prev_tick))
        self.prev_tick = now

        items = [f"FPS: {fps:0.1f}"] + [f"{k}: {v}" for k, v in self.metrics.items()]
        text = "  |  ".join(items)
        cv2.putText(hud, text, (15, int(self.hud_height * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.line(hud, (0, self.hud_height - 1), (self.total_w, self.hud_height - 1), (60, 60, 60), 1)
        return hud

    def _prepare_tile(self, data, target_w, target_h) -> np.ndarray:
        tile = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if data is None or data["img"] is None:
            # cv2.putText(tile, "EMPTY", (self.cell_w // 2 - 25, self.cell_h // 2),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
            return tile

        raw = data["img"]
        # Format normalization
        if len(raw.shape) == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif len(raw.shape) == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        h, w = raw.shape[:2]
        if w != target_w or h != target_h:
            # Resizing
            interp = cv2.INTER_NEAREST if data["is_mask"] else cv2.INTER_AREA
            scale = min(target_w / w, target_h / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            
            resized = cv2.resize(raw, (nw, nh), interpolation=interp)
            
            px, py = (target_w - nw) // 2, (target_h - nh) // 2
            tile[py:py + nh, px:px + nw] = resized
            # print("YES RESIZE...........")
        else:
            # print("NOOOOOOOOOOOOO RESIZE...........")
            tile = raw.copy()

        # print(f"{data["label"]}")

        cv2.rectangle(tile, (0, 0), (target_w, 28), (25, 25, 25), -1)
        cv2.putText(tile, data["label"], (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (target_w - 1, target_h - 1), (60, 60, 60), 1)
        return tile

    def render(self) -> np.ndarray:
        canvas = np.zeros((self.grid_h, self.total_w, 3), dtype=np.uint8)
        
        # Draw Hero View
        if "hero" in self.views:
            canvas[0:self.hero_h, 0:self.hero_w] = self._prepare_tile(
                self.views["hero"], self.hero_w, self.hero_h
            )

        # Draw Aux Views Stacked on Right
        aux_keys = [k for k in self.views.keys() if k != "hero"]
        if aux_keys:
            aux_h = self.grid_h // len(aux_keys)
            for i, key in enumerate(aux_keys):
                y = i * aux_h
                h = (self.grid_h - y) if i == len(aux_keys) - 1 else aux_h
                canvas[y:y + h, self.hero_w:self.total_w] = self._prepare_tile(
                    self.views[key], self.aux_w, h
                )

        return np.vstack([self._render_hud(), canvas])

    def show(self, window_name="Dashboard", wait_ms=1):
        cv2.imshow(window_name, self.render())
        return cv2.waitKey(wait_ms)

    def clear(self):
        self.views.clear()
