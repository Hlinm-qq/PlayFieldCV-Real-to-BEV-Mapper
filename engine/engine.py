from enum import Enum
from ultralytics import YOLO

class EngineType(Enum):
    ULTRALYTICSYOLO = 1
    OTHER = 2

def get_object_classes(model, model_source: EngineType):
    if (model_source == EngineType.ULTRALYTICSYOLO):
        return list(model.names.values())
    return []

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