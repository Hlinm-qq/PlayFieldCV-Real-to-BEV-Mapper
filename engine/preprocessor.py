import cv2
import torch
import numpy as np

class Preprocessor:
    def process(self, raw_frame, target_shape=(640, 640)):
        """
        Transforms frame into tensor
        raw_frame: [H, W, C]
        target_shape: [W, H]
        Returns: the tensor, ratio, padding
        """
        
        # Letterboxing (Resize with Padding)
        shape = raw_frame.shape[:2]  # [H, W]
        
        # Calculate the scaling ratio (r) to fit the image inside the target shape
        r = min(target_shape[0] / shape[1], target_shape[1] / shape[0])
        
        # Compute the new dimensions using that ratio [W, H]
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # Compute the padding needed to reach exactly target shape
        dw = (target_shape[0] - new_unpad[0]) / 2  # Width padding
        dh = (target_shape[1] - new_unpad[1]) / 2  # Height padding

        # Actually resize the image
        if shape[::-1] != new_unpad:  
            im = cv2.resize(raw_frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            im = raw_frame.copy()

        # Add the borders (padding)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # 114 is the standard "gray" value used by YOLO to indicate empty space 
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Color conversion
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Format the tensor
        # OpenCV: [Height, Width, Channels].
        # ONNX/PyTorch: [Channels, Height, Width].
        im = im.transpose((2, 0, 1))

        # Add the "Batch" dimension [1, C, S, S]
        im = np.expand_dims(im, axis=0)
        
        # Normalization + memory arrangement
        im = np.ascontiguousarray(im, dtype=np.float32) / 255.0

        return torch.from_numpy(im), r, dw, dh
