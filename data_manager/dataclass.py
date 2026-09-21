from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, NamedTuple, Tuple
import numpy as np

@dataclass
class PlayerDetection:
    track_id: int
    bbox: BBox
    src_coords: Vector2D
    map_coords: Vector2D
    confidence: float = 0.0

@dataclass
class FrameMetrics:
    detection_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    total_pipeline_time_ms: float = 0.0

@dataclass
class FrameResult:
    # frame_idx: int
    players: list[PlayerDetection] = field(default_factory=list)
    court_corners: np.ndarray | None = None
    dst_points: np.ndarray | None = None
    homography_matrix: np.ndarray | None = None 
    # metrics: FrameMetrics = field(default_factory=FrameMetrics)
    floor_mask: np.ndarray | None = None

class Vector2D(NamedTuple):
    x: float
    y: float
    # var = Vector2D(x=450.5, y=1020.0)
    # print(foot_pos.x)  # 450.5

@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Vector2D:
        """Returns the geometric center for ReID cropping."""
        return Vector2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Vector2D:
        """Returns the player's feet location for 2D court mapping."""
        return Vector2D((self.x1 + self.x2) / 2.0, self.y2)
        
    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Returns integer coordinates for OpenCV drawing functions."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

