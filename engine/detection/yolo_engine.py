import numpy as np
from pathlib import Path
from enum import Enum, IntFlag, auto

from ultralytics import YOLO

class EngineType(Enum):
    ULTRALYTICSYOLO = 1
    OTHER = 2

class BoxFormat(IntFlag):
    NONE = 0
    XYXY = auto()      # 1
    XYWH = auto()      # 2
    CLASSES = auto()   # 4
    CONFIDENCE = auto()# 8
    ID = auto()
    KEYPOINT = auto()
    ALL = XYXY | XYWH | CLASSES | CONFIDENCE | ID

class ModelType(Enum):
    STANDARD = 1
    TRACKING = 2

class EngineManager:
    def __init__(
            self, 
            type: EngineType, 
            weight_path, 
            using_raw=False, 
            device=None, 
            use_tracking=False, 
            is_video=False, 
            id_filter=None,
            imgsz=1280,
            conf=0.1,
            iou=0.5,         # NMS overlap threshold
            tracker_config="volleyball_tracker.yaml"
        ):
        
        self.engine_type: EngineType = type
        self.weight_path = weight_path
        self.using_raw = using_raw
        self.device = device 

        self.use_tracking = use_tracking
        self.is_video = is_video

        self.model = None
        self.class_id_filter = id_filter

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker_config = self._resolve_tracker_config(tracker_config)

        self.load_model(self.using_raw)
        if self.device:
            self.model.to(self.device)

        # # Warmup
        # dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # self.model(dummy_frame)

    def _resolve_tracker_config(self, config_target: str) -> str:
        target = Path(config_target)

        if target.is_file():
            return str(target.resolve())

        project_root = Path(__file__).resolve().parent.parent.parent
        local_config = project_root / "config" / target.name

        if local_config.is_file():
            return str(local_config.resolve())

        print(f"WARNING: Custom config '{config_target}' not found. Falling back to default 'bytetrack.yaml'.")
        return "bytetrack.yaml"

    def load_model(self, is_raw=False):
        """
        Load model to be used as the Inference Engine (Brain)
        """
        if self.engine_type == EngineType.ULTRALYTICSYOLO:
            if not is_raw:
                self.model = YOLO(self.weight_path).model
                self.model.eval()
            else:
                self.model = YOLO(self.weight_path) 

        return self.model
    
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
                keypoints = result.keypoints
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

                    # Extract Tracking IDs
                    if BoxFormat.ID in desired_flags:
                        if boxes.id is not None:
                            entry['id'] = boxes.id.int().cpu().numpy()
                        else:
                            entry['id'] = [-1] * len(boxes)

                    if BoxFormat.KEYPOINT in desired_flags:
                        if keypoints is not None:
                            # x, y coordinates for all 17 points
                            entry['keypoints_xy'] = keypoints.xy.cpu().numpy() 
                            # confidence for all 17 points
                            entry['keypoints_conf'] = keypoints.conf.cpu().numpy()

        return entry

    def detect(self, frame):
        if self.use_tracking:
            outputs = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.class_id_filter,
                verbose=False
            )
        else:
            outputs = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=self.class_id_filter,
                verbose=False
            )
        results = self.get_output(outputs, BoxFormat.XYXY | BoxFormat.CONFIDENCE | BoxFormat.ID)

        return results
          