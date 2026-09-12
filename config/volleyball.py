import numpy as np

# Standard bird's-eye view dimensions (e.g., in centimeters or pixels)
COURT_WIDTH = 900
COURT_LENGTH = 1800

# Mapping of different court layouts or camera angles to their 2D map coordinates
COURT_ZONES = {
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    "1": {
        "name": "Full Court (Outer Corners)",
        "dst_pts": np.array([[0,0], [1800,0], [1800,900], [0,900]], dtype=np.float32)
    },
    "2": {
        "name": "Left Half (Baseline to Center Line)",
        "dst_pts": np.array([[0,0], [900,0], [900,900], [0,900]], dtype=np.float32)
    },
    "3": {
        "name": "Left Attack Zone (Attack Line to Center Line)",
        "dst_pts": np.array([[600,0], [900,0], [900,900], [600,900]], dtype=np.float32)
    },
    "4": {
        "name": "Right Attack Zone (Center Line to Right Attack Line)",
        "dst_pts": np.array([[900,0], [1200,0], [1200,900], [900,900]], dtype=np.float32)
    },
    "5": {
        "name": "Left Back Zone (Left baseline to left attack line)",
        "dst_pts": np.array([[0, 0], [600, 0], [600, 900], [0, 900]], dtype=np.float32)
    },
    "6": {
        "name": "Right Back Zone (Right attack line to right baseline)",
        "dst_pts": np.array([[1200, 0], [1800, 0], [1800, 900], [1200, 900]], dtype=np.float32)
    }
}

# Fallback points for testing without the PointPicker
DEFAULT_SRC_POINTS = np.float32([[147, 329], [291, 333], [269, 483], [81, 474]])