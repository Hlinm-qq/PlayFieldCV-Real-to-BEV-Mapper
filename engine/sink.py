import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from utils.utils import *

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box [x_min, y_min, x_max, y_max] 
    Returns the measurement vector z: [u, v, s, r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h       # scale is area
    r = w / float(h) # aspect ratio
    
    # Return as a 4x1 numpy column vector
    return np.array([u, v, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    """
    Takes the Kalman state vector x: [u, v, s, r, u_dot, v_dot, s_dot]
    Returns a standard bounding box [x_min, y_min, x_max, y_max]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    
    x_min = x[0] - w / 2.0
    y_min = x[1] - h / 2.0
    x_max = x[0] + w / 2.0
    y_max = x[1] + h / 2.0
    
    return np.array([x_min, y_min, x_max, y_max]).reshape(4)

class Track:
    def __init__(self, detection, track_id):
        self.id = track_id
        self.hits = 1           
        self.time_since_update = 0 
        
        # Initialize the Kalman Filter (7 state variables, 4 observed variables)
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        
        # State Transition Matrix (F)
        # The "physics model"
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        
        # Measurement Function Matrix (H)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        
        # Measurement Noise Matrix (R): 4x4
        # We trust the x/y coordinates a lot, but we trust the bounding box size (scale/ratio) less.
        self.kf.R[2:, 2:] *= 10.
        
        # State Covariance Matrix (P): 7x7
        # Represents initial uncertainty. 
        self.kf.P *= 10.
        # Uncertainty (1000) for the initial velocities
        self.kf.P[4:, 4:] *= 1000. 
        
        # Process Noise Matrix (Q): 7x7
        # How much do we trust the Kalman Filter's own predictions?
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # The state (x): 7x1
        # Initialize with the very first detection, 
        # convert the raw box to [u, v, s, r] and the velocities default to 0
        self.kf.x[:4] = convert_bbox_to_z(detection)

    def predict(self):
        """Advances the state vector based on current velocity."""
        self.kf.predict()
        self.time_since_update += 1
        
        # Return the prediction as a standard bounding box [x_min, y_min, x_max, y_max]
        return convert_x_to_bbox(self.kf.x)

    def update(self, detection):
        """Corrects the state vector based on actual AI observations."""
        # Convert the incoming YOLO box into the [u, v, s, r] measurement format
        z = convert_bbox_to_z(detection)
        
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1

class SORTTracker:
    """
    The Orchestrator that manages all active Tracks and assigns IDs.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age   # Frames to wait before deleting a lost track
        self.min_hits = min_hits # Frames a track must exist before it's "official"
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1

    def update(self, detections):
        """
        detections: list of dicts: [{'box': [x1, y1, x2, y2], 'score': 0.9, 'class_id': 0}, ...]
        returns: list of dicts with 'track_id' added and smoothed 'box' coordinates.
        """
        self.frame_count += 1
        
        # Get predictions from existing tracks
        predicted_boxes = []
        to_del = []
        for i, track in enumerate(self.tracks):
            pred = track.predict()
            if np.any(np.isnan(pred)): # Safety check for exploding gradients/math
                to_del.append(i)
            else:
                predicted_boxes.append(pred)
                
        for i in reversed(to_del):
            self.tracks.pop(i)

        # Build the Cost Matrix (IoU)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, trk_box in enumerate(predicted_boxes):
            for d, det in enumerate(detections):
                cost_matrix[t, d] = -calculate_iou(trk_box, det['box'])

        # Hungarian Algorithm Assignment
        matched_indices = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if -cost_matrix[r, c] < self.iou_threshold:
                    continue # Reject matches that overlap too little
                matched_indices.append((r, c))
                if r in unmatched_tracks: unmatched_tracks.remove(r)
                if c in unmatched_detections: unmatched_detections.remove(c)

        # Update Matched Tracks
        for trk_idx, det_idx in matched_indices:
            self.tracks[trk_idx].update(detections[det_idx]['box'])

        # Create New Tracks
        for d in unmatched_detections:
            self.tracks.append(Track(detections[d]['box'], self.next_id))
            self.next_id += 1

        # Delete Dead Tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Format the Output
        active_tracks = []
        for t in self.tracks:
            # Only return tracks that have hit the "min_hits" threshold (not ghosts)
            if t.time_since_update < 1 and (t.hits >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = convert_x_to_bbox(t.kf.x)
                active_tracks.append({
                    "track_id": t.id,
                    "box": bbox,
                })
                
        return active_tracks







