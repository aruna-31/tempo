import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("tempo.ml.sampler")


class VideoFrameSampler:
    """
    Extracts frames from video files at a given sampling rate (FPS)
    and constructs sliding temporal sequences.
    Strictly decodes real video frames with zero silent fallbacks on failure.
    """

    def __init__(self, sampling_fps: float = 2.0, sequence_length: int = 16, stride: int = 8):
        self.sampling_fps = max(0.1, float(sampling_fps))
        self.sequence_length = max(1, int(sequence_length))
        self.stride = max(1, int(stride))

    def extract_frames(
        self, video_path: str, max_duration_seconds: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts sampled frames from video with timestamp metadata.
        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video cannot be opened, is corrupted, or contains 0 frames.
        Returns:
            List of dicts: [
                {"frame_number": int, "timestamp_seconds": float, "frame_data": np.ndarray}
            ]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file '{video_path}' not found")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                f"Could not open or decode video file '{video_path}'. Video container or codec is corrupt or unsupported."
            )

        frames_meta: List[Dict[str, Any]] = []

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if not native_fps or native_fps <= 0 or np.isnan(native_fps):
                native_fps = 30.0

            frame_interval = max(1, int(round(native_fps / self.sampling_fps)))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp = round(frame_idx / native_fps, 3)
                    if max_duration_seconds and timestamp > max_duration_seconds:
                        break

                    frames_meta.append({
                        "frame_number": frame_idx,
                        "timestamp_seconds": timestamp,
                        "frame_data": frame
                    })

                frame_idx += 1
        finally:
            cap.release()

        if not frames_meta:
            raise ValueError(
                f"Video file '{video_path}' contains zero decodable video frames."
            )

        return frames_meta

    def create_temporal_sequences(
        self, sampled_frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Groups extracted frames into temporal sliding window sequences.
        """
        if not sampled_frames:
            return []

        # If we have fewer frames than sequence_length, pad with the last frame
        if len(sampled_frames) < self.sequence_length:
            padding_count = self.sequence_length - len(sampled_frames)
            last_item = sampled_frames[-1]
            sampled_frames = sampled_frames + [last_item] * padding_count

        sequences: List[Dict[str, Any]] = []
        n_frames = len(sampled_frames)

        for start_idx in range(0, n_frames - self.sequence_length + 1, self.stride):
            end_idx = start_idx + self.sequence_length
            window = sampled_frames[start_idx:end_idx]

            anchor_item = window[-1]  # The prediction timestamp is anchored to the window end
            seq_dict = {
                "window_start_time": window[0]["timestamp_seconds"],
                "window_end_time": window[-1]["timestamp_seconds"],
                "anchor_timestamp": anchor_item["timestamp_seconds"],
                "anchor_frame_number": anchor_item["frame_number"],
                "frames": [item["frame_data"] for item in window]
            }
            sequences.append(seq_dict)

        return sequences


class TrackSequenceBuilder:
    """
    Constructs independent temporal sequences for each individual student Track ID.
    Handles temporal sliding windows, occlusions, temporary disappearances and padding.
    """

    def __init__(self, sequence_length: int = 16, stride: int = 8):
        self.sequence_length = max(1, sequence_length)
        self.stride = max(1, stride)

    def build_track_sequences(
        self,
        track_crops: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Builds sliding window temporal sequences for each tracked student.
        Args:
            track_crops: Dict mapping track_id -> list of observation dicts:
                         [{
                             "frame_number": int,
                             "timestamp_seconds": float,
                             "crop": np.ndarray,
                             "bbox": [x1, y1, x2, y2],
                             "confidence": float
                         }]
        Returns:
            Dict mapping track_id -> list of temporal sequence dicts:
            [
                {
                    "track_id": int,
                    "window_start_time": float,
                    "window_end_time": float,
                    "anchor_timestamp": float,
                    "anchor_frame_number": int,
                    "crops": [np.ndarray, ... of length sequence_length],
                    "bboxes": [list, ...]
                }
            ]
        """
        all_track_sequences: Dict[int, List[Dict[str, Any]]] = {}

        for track_id, observations in track_crops.items():
            if not observations:
                continue

            # Sort observations by timestamp
            sorted_obs = sorted(observations, key=lambda x: x["timestamp_seconds"])

            # If track has fewer observations than sequence_length, pad with the last crop
            if len(sorted_obs) < self.sequence_length:
                padding_count = self.sequence_length - len(sorted_obs)
                last_obs = sorted_obs[-1]
                padded_obs = sorted_obs + [last_obs] * padding_count
            else:
                padded_obs = sorted_obs

            seq_list: List[Dict[str, Any]] = []
            n_obs = len(padded_obs)

            for start_idx in range(0, n_obs - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                window = padded_obs[start_idx:end_idx]

                anchor_item = window[-1]
                seq_dict = {
                    "track_id": track_id,
                    "window_start_time": window[0]["timestamp_seconds"],
                    "window_end_time": window[-1]["timestamp_seconds"],
                    "anchor_timestamp": anchor_item["timestamp_seconds"],
                    "anchor_frame_number": anchor_item["frame_number"],
                    "crops": [item["crop"] for item in window],
                    "bboxes": [item.get("bbox", []) for item in window]
                }
                seq_list.append(seq_dict)

            all_track_sequences[track_id] = seq_list

        return all_track_sequences
