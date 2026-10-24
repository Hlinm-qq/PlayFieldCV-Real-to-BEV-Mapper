import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

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
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
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

    def add(self, name: str, frame: np.ndarray):
        """Buffer a stage output by name."""
        if not self.enabled or frame is None:
            return
        self.frames[name] = frame.copy()

    def _compute_cell_dimensions(self, active_rows: int) -> tuple[int, int]:
        """Derive individual cell dimensions from the canvas limit."""
        cell_w = self.max_w // self.cols
        cell_h = self.max_h // active_rows
        return cell_w, cell_h

    def _prepare_tile(self, name: str, img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
        tile = img.copy()

        # Normalize types & bit depth
        if tile.dtype == bool:
            tile = (tile.astype(np.uint8)) * 255
        elif np.issubdtype(tile.dtype, np.floating):
            tile = np.clip(tile * 255 if tile.max() <= 1.0 else tile, 0, 255).astype(np.uint8)

        # Standardize to 3 channels (BGR)
        if len(tile.shape) == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        elif len(tile.shape) == 3 and tile.shape[2] == 4:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)

        # Resize
        h, w = tile.shape[:2]
        if self.keep_aspect:
            scale = min(cell_w / w, cell_h / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_AREA)

            # Center resized image onto a black cell canvas
            canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            pad_x = (cell_w - nw) // 2
            pad_y = (cell_h - nh) // 2
            canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
            tile = canvas
        else:
            tile = cv2.resize(tile, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        # Stage Title Overlay
        cv2.rectangle(tile, (0, 0), (cell_w, 24), (25, 25, 25), -1)
        cv2.putText(
            tile, name, (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
        )
        return tile

    def render_grid(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((self.max_h, self.max_w, 3), dtype=np.uint8)

        # Determine effective rows
        n = len(self.frames)
        if self.rows is not None:
            active_rows = self.rows
        else:
            active_rows = (n + self.cols - 1) // self.cols

        cell_w, cell_h = self._compute_cell_dimensions(active_rows)

        # Prepare tiles
        tiles = [
            self._prepare_tile(name, frame, cell_w, cell_h)
            for name, frame in self.frames.items()
        ]

        # Handle overflow or underflow
        total_slots = active_rows * self.cols
        if len(tiles) > total_slots:
            tiles = tiles[:total_slots]  # Truncate if explicit rows*cols is exceeded
        elif len(tiles) < total_slots:
            blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            tiles.extend([blank] * (total_slots - len(tiles)))

        # Assemble matrix
        rows = [
            np.hstack(tiles[r * self.cols : (r + 1) * self.cols])
            for r in range(active_rows)
        ]
        return np.vstack(rows)

    def show(self, window_name="Pipeline Debugger", wait_ms=1):
        if not self.enabled or not self.frames:
            return
        cv2.imshow(window_name, self.render_grid())
        return cv2.waitKey(wait_ms)

    def clear(self):
        self.frames.clear()