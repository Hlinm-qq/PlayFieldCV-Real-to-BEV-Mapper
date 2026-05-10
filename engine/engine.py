from enum import Enum
from ultralytics import YOLO
from enum import IntFlag, auto

class EngineType(Enum):
    ULTRALYTICSYOLO = 1
    OTHER = 2

class BoxFormat(IntFlag):
    NONE = 0
    XYXY = auto()      # 1
    XYWH = auto()      # 2
    CLASSES = auto()   # 4
    CONFIDENCE = auto()# 8
    ALL = XYXY | XYWH | CLASSES | CONFIDENCE 

class EngineManager:
    def __init__(self, type: EngineType, weight_path, using_raw=False):
        self.weight_path = weight_path
        self.using_raw = using_raw
        self.engine_type: EngineType = type

    def load_model(self, type: EngineType, is_raw=False):
        """
        Load model to be used as the Inference Engine (Brain)
        """
        model = None
        if type == EngineType.ULTRALYTICSYOLO:
            if is_raw:
                model = YOLO(self.weight_path).model
                model.eval()
            else:
                model = YOLO(self.weight_path) 

        return model
    
    def get_output(self, model_output, desired_flags: BoxFormat):
        """
        Parses raw model output based on selected flags.
        """
        processed_results = []
        entry = {}
        results = model_output

        if self.engine_type == EngineType.ULTRALYTICSYOLO:
            # print("results")
            # print(results)
            for result in results:
                boxes = result.boxes
                # print("boxes")
                # print(boxes)
                if boxes is not None and len(boxes) > 0:
                    # Bitwise checks to see what to include

                    if BoxFormat.XYXY in desired_flags:
                        entry['xyxy'] = boxes.xyxy.cpu().numpy().astype(int)

                    if BoxFormat.XYWH in desired_flags:
                        entry['xywh'] = boxes.xywh.cpu().numpy().astype(int)

                    if BoxFormat.CLASSES in desired_flags:
                        names = [result.names[id.item()] for id in boxes.cls.int().cpu().numpy()] # class name of each box
                        entry['class'] = names

                    if BoxFormat.CONFIDENCE in desired_flags:
                        entry['confidence'] = boxes.conf.cpu().numpy()
           
        return entry
