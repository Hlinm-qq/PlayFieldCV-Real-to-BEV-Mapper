import cv2
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