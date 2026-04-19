import cv2

class OutputManager:
    def __init__(self):
        self.display_cell = None
        self.is_notebook = -1

        self.display = None
        self.Image = None
        self.clear_output = None

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
