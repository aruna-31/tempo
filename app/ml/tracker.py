import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from app.ml.appearance import appearance_distance

logger = logging.getLogger("tempo.ml.tracker")


class TrackState(str, Enum):
    NEW = "NEW"
    TRACKED = "TRACKED"
    LOST = "LOST"
    REMOVED = "REMOVED"


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes in image space [x1, y1, x2, y2].
    State: [x_center, y_center, aspect_ratio (w/h), height, vx, vy, va, vh]
    """

    def __init__(self, bbox: np.ndarray):
        # [x1, y1, x2, y2]
        self.dim_x = 8
        self.dim_z = 4

        # Transition matrix F
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H
        self.H = np.eye(self.dim_z, self.dim_x)

        # Covariance matrices
        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty for velocities

        self.R = np.eye(self.dim_z) * 1.0  # measurement noise
        self.Q = np.eye(self.dim_x) * 0.01  # process noise
        self.Q[4:, 4:] *= 0.01

        # Initial state
        self.x = np.zeros((self.dim_x, 1))
        self.x[:4] = self._bbox_to_z(bbox)

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Converts [x1, y1, x2, y2] to [x_c, y_c, s, h]^T where s = w/h."""
        w = max(1e-2, bbox[2] - bbox[0])
        h = max(1e-2, bbox[3] - bbox[1])
        x_c = bbox[0] + w / 2.0
        y_c = bbox[1] + h / 2.0
        s = w / h
        return np.array([[x_c], [y_c], [s], [h]])

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """Converts [x_c, y_c, s, h]^T to [x1, y1, x2, y2]."""
        x_c, y_c, s, h = z[0, 0], z[1, 0], z[2, 0], z[3, 0]
        h = max(1.0, h)
        w = max(1.0, s * h)
        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Predicts the next state and returns the predicted bounding box."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return KalmanBoxTracker._z_to_bbox(self.x[:4])

    def update(self, bbox: np.ndarray) -> None:
        """Updates the state with a new observation [x1, y1, x2, y2]."""
        z = self._bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)  # innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


class STrack:
    """
    Single object track representing a temporary student track in a session.
    """
    _count = 0

    def __init__(self, bbox: np.ndarray, score: float, frame_idx: int, timestamp: float, appearance: Optional[np.ndarray] = None):
        # [x1, y1, x2, y2]
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = float(score)
        self.kalman_filter: Optional[KalmanBoxTracker] = None
        self.track_id = 0
        self.state = TrackState.NEW
        self.is_activated = False

        self.start_frame = frame_idx
        self.end_frame = frame_idx
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.time_since_update = 0
        self.hits = 1
        self.appearance = appearance

        # Trajectory history: list of dicts with frame, timestamp, bbox, confidence
        self.history: List[Dict[str, Any]] = [{
            "frame": frame_idx,
            "timestamp": timestamp,
            "bbox": [round(float(c), 2) for c in self.bbox],
            "confidence": round(float(self.score), 4)
        }]

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls) -> None:
        cls._count = 0

    def activate(self, frame_idx: int, timestamp: float) -> None:
        """Activates a new track."""
        self.kalman_filter = KalmanBoxTracker(self.bbox)
        self.track_id = self.next_id()
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.start_frame = frame_idx
        self.end_frame = frame_idx
        self.time_since_update = 0
        self.hits = 1

    def re_activate(self, new_track: "STrack", frame_idx: int, timestamp: float, new_id: bool = False) -> None:
        """Re-activates a lost track with a new observation."""
        if self.kalman_filter is not None:
            self.kalman_filter.update(new_track.bbox)
            self.bbox = KalmanBoxTracker._z_to_bbox(self.kalman_filter.x[:4])
        else:
            self.bbox = new_track.bbox
            self.kalman_filter = KalmanBoxTracker(self.bbox)

        self.score = new_track.score
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.end_frame = frame_idx
        self.time_since_update = 0
        self.hits += 1
        if new_track.appearance is not None:
            self.appearance = new_track.appearance
        if new_id:
            self.track_id = self.next_id()

        self.history.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "bbox": [round(float(c), 2) for c in self.bbox],
            "confidence": round(float(self.score), 4)
        })

    def update(self, new_track: "STrack", frame_idx: int, timestamp: float) -> None:
        """Updates track with matching detection."""
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.end_frame = frame_idx
        self.time_since_update = 0
        self.hits += 1
        if new_track.appearance is not None:
            self.appearance = new_track.appearance

        if self.kalman_filter is not None:
            self.kalman_filter.update(new_track.bbox)
            self.bbox = KalmanBoxTracker._z_to_bbox(self.kalman_filter.x[:4])
        else:
            self.bbox = new_track.bbox

        self.score = new_track.score
        self.state = TrackState.TRACKED
        self.is_activated = True

        self.history.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "bbox": [round(float(c), 2) for c in self.bbox],
            "confidence": round(float(self.score), 4)
        })

    def predict(self) -> np.ndarray:
        """Predicts the next bounding box position."""
        if self.kalman_filter is not None:
            self.bbox = self.kalman_filter.predict()
        return self.bbox

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


def compute_iou_matrix(tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
    """Computes IoU cost matrix between tracks and detections."""
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float32)

    boxes_a = np.array([t.bbox for t in tracks], dtype=np.float32)
    boxes_b = np.array([d.bbox for d in detections], dtype=np.float32)

    # boxes: [N, 4] -> x1, y1, x2, y2
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    ious = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = area_a[i] + area_b[j] - inter
            ious[i, j] = inter / union if union > 0 else 0.0

    return ious


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Linear sum assignment (Hungarian matching) on cost matrix."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int]] = []
    unmatched_a: List[int] = list(range(cost_matrix.shape[0]))
    unmatched_b: List[int] = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b


class ByteTracker:
    """
    ByteTrack Multi-Object Tracking (MOT) Implementation.
    Maintains session-specific student Track IDs across temporary occlusions and disappearances.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        match_thresh_second: float = 0.5,
        max_time_lost_frames: int = 30,
        center_distance_gate: Optional[float] = None,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.match_thresh_second = match_thresh_second
        self.max_time_lost_frames = max_time_lost_frames
        self.center_distance_gate = center_distance_gate

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.all_completed_tracks: List[STrack] = []

        self.frame_id = 0
        STrack.reset_counter()

    def _association_cost(self, tracks: List[STrack], detections: List[STrack]) -> np.ndarray:
        iou_cost = 1.0 - compute_iou_matrix(tracks, detections)
        if self.center_distance_gate is None or iou_cost.size == 0:
            return iou_cost

        costs = iou_cost.copy()
        for track_index, track in enumerate(tracks):
            tx = (track.bbox[0] + track.bbox[2]) / 2.0
            ty = (track.bbox[1] + track.bbox[3]) / 2.0
            scale = max(1.0, max(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]))
            for detection_index, detection in enumerate(detections):
                dx = (detection.bbox[0] + detection.bbox[2]) / 2.0
                dy = (detection.bbox[1] + detection.bbox[3]) / 2.0
                distance = float(np.hypot(dx - tx, dy - ty) / scale)
                if distance > self.center_distance_gate:
                    costs[track_index, detection_index] = 1.01
                else:
                    distance_cost = min(1.0, distance / self.center_distance_gate)
                    costs[track_index, detection_index] = 0.7 * costs[track_index, detection_index] + 0.3 * distance_cost
        return costs

    def reset(self) -> None:
        """Resets tracker state for a new video session."""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.all_completed_tracks.clear()
        self.frame_id = 0
        STrack.reset_counter()

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_seconds: float
    ) -> List[STrack]:
        """
        Updates tracks with detections for the current frame.
        Args:
            detections: List of dicts with {"bbox": [x1, y1, x2, y2], "confidence": float}
            frame_idx: int frame index
            timestamp_seconds: float timestamp
        Returns:
            List of currently active STrack objects.
        """
        self.frame_id = frame_idx

        # 1. Separate detections into high-score and low-score pools
        det_high: List[STrack] = []
        det_low: List[STrack] = []

        for det in detections:
            bbox = np.array(det["bbox"], dtype=np.float32)
            score = float(det.get("confidence", 1.0))
            strack = STrack(bbox, score, frame_idx, timestamp_seconds, det.get("appearance"))
            if score >= self.track_thresh:
                det_high.append(strack)
            else:
                det_low.append(strack)

        # 2. Predict current locations of existing tracks with Kalman filter
        track_pool = self.tracked_stracks + self.lost_stracks
        for track in track_pool:
            track.predict()

        # 3. First association: match tracked tracks with high score detections
        cost_matrix = self._association_cost(self.tracked_stracks, det_high)
        matches_a, u_track_a, u_det_high = linear_assignment(cost_matrix, thresh=self.match_thresh)

        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []

        for itracked, idet in matches_a:
            track = self.tracked_stracks[itracked]
            det = det_high[idet]
            track.update(det, frame_idx, timestamp_seconds)
            activated_stracks.append(track)

        # 4. Second association: match remaining tracks with low score detections
        r_tracked_stracks = [self.tracked_stracks[i] for i in u_track_a]
        cost_matrix_second = self._association_cost(r_tracked_stracks, det_low)
        matches_b, u_track_b, _ = linear_assignment(cost_matrix_second, thresh=self.match_thresh_second)

        for itracked, idet in matches_b:
            track = r_tracked_stracks[itracked]
            det = det_low[idet]
            track.update(det, frame_idx, timestamp_seconds)
            activated_stracks.append(track)

        # Unmatched tracked tracks become Lost
        new_lost_stracks: List[STrack] = []
        for i in u_track_b:
            track = r_tracked_stracks[i]
            track.mark_lost()
            new_lost_stracks.append(track)

        # 5. Association with lost tracks for remaining high score detections
        u_det_high_stracks = [det_high[i] for i in u_det_high]
        cost_matrix_lost = self._association_cost(self.lost_stracks, u_det_high_stracks)
        matches_lost, u_lost, u_det_final = linear_assignment(cost_matrix_lost, thresh=self.match_thresh)

        for ilost, idet in matches_lost:
            track = self.lost_stracks[ilost]
            det = u_det_high_stracks[idet]
            track.re_activate(det, frame_idx, timestamp_seconds)
            refind_stracks.append(track)

        # 6. Initialize new tracks from unmatched high score detections
        for i in u_det_final:
            track = u_det_high_stracks[i]
            track.activate(frame_idx, timestamp_seconds)
            activated_stracks.append(track)

        # 7. Update lost tracks and remove dead tracks
        for i in u_lost:
            track = self.lost_stracks[i]
            track.time_since_update += 1
            if track.time_since_update > self.max_time_lost_frames:
                track.mark_removed()
                self.removed_stracks.append(track)
                self.all_completed_tracks.append(track)

        # Update tracking lists
        self.tracked_stracks = [t for t in (activated_stracks + refind_stracks) if t.state == TrackState.TRACKED]
        unmatched_lost = [self.lost_stracks[i] for i in u_lost if self.lost_stracks[i].state == TrackState.LOST]
        self.lost_stracks = unmatched_lost + new_lost_stracks

        # Combine all active tracks for current frame
        current_active = [t for t in self.tracked_stracks if t.is_activated]
        return current_active

    def get_all_session_tracks(self) -> List[STrack]:
        """Returns all tracks (active, lost, removed) seen during the session."""
        all_tracks_dict = {}
        for t in self.all_completed_tracks + self.removed_stracks + self.lost_stracks + self.tracked_stracks:
            if t.track_id > 0 and t.track_id not in all_tracks_dict:
                all_tracks_dict[t.track_id] = t
        return sorted(list(all_tracks_dict.values()), key=lambda x: x.track_id)


class DenseByteTracker(ByteTracker):
    """ByteTrack variant with confirmation and center-distance gating."""

    def __init__(
        self,
        track_thresh: float = 0.30,
        match_thresh: float = 0.80,
        match_thresh_second: float = 0.40,
        max_time_lost_frames: int = 12,
        confirmation_hits: int = 2,
        center_distance_gate: float = 2.5,
        appearance_weight: float = 0.25,
        appearance_gate: float = 0.65,
    ):
        super().__init__(
            track_thresh,
            match_thresh,
            match_thresh_second,
            max_time_lost_frames,
            center_distance_gate,
        )
        self.confirmation_hits = max(1, int(confirmation_hits))
        self.center_distance_gate = float(center_distance_gate)
        self.appearance_weight = max(0.0, min(1.0, float(appearance_weight)))
        self.appearance_gate = float(appearance_gate)

    def _association_cost(self, tracks, detections):
        costs = super()._association_cost(tracks, detections)
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                distance = appearance_distance(track.appearance, detection.appearance)
                if distance > self.appearance_gate:
                    costs[track_index, detection_index] = 1.01
                elif costs[track_index, detection_index] <= 1.0:
                    costs[track_index, detection_index] = (
                        (1.0 - self.appearance_weight) * costs[track_index, detection_index]
                        + self.appearance_weight * distance
                    )
        return costs

    def update(self, detections, frame_idx, timestamp_seconds):
        active = super().update(detections, frame_idx, timestamp_seconds)
        confirmed = []
        for track in active:
            if track.hits >= self.confirmation_hits:
                confirmed.append(track)
        return confirmed

    def get_confirmed_session_tracks(self):
        return [track for track in super().get_all_session_tracks() if track.hits >= self.confirmation_hits]

    def get_all_observed_tracks(self):
        """Return confirmed and unconfirmed tracks for audit diagnostics."""
        return super().get_all_session_tracks()

    def get_all_session_tracks(self):
        """Expose only confirmed tracks as public session IDs."""
        return self.get_confirmed_session_tracks()
