import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from app.core.config import settings

logger = logging.getLogger("tempo.ml.annotator")

# Distinct color palette (BGR) for student bounding boxes
TRACK_COLORS = [
    (255, 128, 0),    # Vivid Blue
    (0, 200, 100),    # Vivid Green
    (0, 140, 255),    # Vivid Orange
    (220, 50, 220),   # Purple/Magenta
    (0, 225, 255),    # Yellow
    (50, 205, 50),    # Lime
    (255, 69, 0),     # Deep Red-Orange
    (180, 105, 255),  # Pink
]


class VideoAnnotator:
    """
    Renders annotated output MP4 videos with:
    - Color-coded student track bounding boxes
    - Session-specific Track IDs and Student Labels
    - Temporal Behaviour Classification labels and confidence percentages
    - Class statistics overlay header
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or settings.OUTPUT_VIDEO_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def render_annotated_video(
        self,
        video_path: str,
        tracks: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Renders annotated video overlaying tracks and temporal predictions.
        Args:
            video_path: Path to original input video file
            tracks: List of student track records with bounding_box_history
            predictions: List of behaviour predictions with timestamp_seconds and track_id
            output_filename: Optional name for output MP4 file
        Returns:
            str: Path to generated annotated video file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video '{video_path}' not found for annotation rendering.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file '{video_path}' for annotation rendering.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0

        if not output_filename:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_annotated.mp4"

        output_path = os.path.join(self.output_dir, output_filename)

        # Build fast lookup map for frame annotations
        # frame_idx -> list of {track_id, bbox, confidence, behaviour, label}
        frame_annotation_map: Dict[int, List[Dict[str, Any]]] = {}

        # Track history mapping
        for track in tracks:
            t_id = track.get("track_id", 1)
            for h in track.get("bounding_box_history", []):
                f_idx = h.get("frame", 0)
                if f_idx not in frame_annotation_map:
                    frame_annotation_map[f_idx] = []
                frame_annotation_map[f_idx].append({
                    "track_id": t_id,
                    "bbox": h.get("bbox", [0, 0, 0, 0]),
                    "confidence": h.get("confidence", 1.0),
                    "behaviour": "Analyzing...",
                    "pred_conf": None
                })

        # Augment with temporal behaviour predictions
        for pred in predictions:
            t_id = pred.get("track_id", 1)
            f_num = pred.get("frame_number", 0)
            b_type = pred.get("behaviour_type", "")
            conf = pred.get("confidence", 0.0)

            # Match to nearest frame annotations in frame_annotation_map
            for offset in range(-15, 16):
                target_f = f_num + offset
                if target_f in frame_annotation_map:
                    for ann in frame_annotation_map[target_f]:
                        if ann["track_id"] == t_id:
                            ann["behaviour"] = b_type
                            ann["pred_conf"] = conf

        # Browsers only play H.264-in-MP4; cv2's FMP4/mp4v output is not playable.
        # Write with mp4v (always available), then transcode to H.264 with the
        # bundled ffmpeg binary (imageio-ffmpeg, pure pip - no system install).
        play_path = output_path  # final browser-playable file
        writer_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(writer_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            # Fallback codec if mp4v is not supported on host OS
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer_path = output_path + ".tmp.avi"
            writer = cv2.VideoWriter(writer_path, fourcc, fps, (width, height))

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Draw top status banner
                banner_h = 36
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, banner_h), (25, 25, 30), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                t_sec = frame_idx / fps
                header_text = f"TEMPO AI | Time: {t_sec:05.1f}s | Frame: {frame_idx} | Tracks: {len(tracks)}"
                cv2.putText(
                    frame, header_text, (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
                )

                # Draw track annotations
                anns = frame_annotation_map.get(frame_idx, [])
                for ann in anns:
                    t_id = ann["track_id"]
                    color = TRACK_COLORS[(t_id - 1) % len(TRACK_COLORS)]
                    bx1, by1, bx2, by2 = [int(round(c)) for c in ann["bbox"]]
                    bx1 = max(0, min(width - 1, bx1))
                    by1 = max(0, min(height - 1, by1))
                    bx2 = max(0, min(width - 1, bx2))
                    by2 = max(0, min(height - 1, by2))

                    if bx2 > bx1 and by2 > by1:
                        # Draw bounding box
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                        # Label badge
                        beh = ann["behaviour"]
                        conf_str = f" ({int(ann['pred_conf'] * 100)}%)" if ann.get("pred_conf") is not None else ""
                        label_text = f"Student {t_id:02d}: {beh}{conf_str}"

                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        badge_y1 = max(0, by1 - th - 8)
                        badge_y2 = by1
                        badge_x2 = min(width, bx1 + tw + 10)

                        cv2.rectangle(frame, (bx1, badge_y1), (badge_x2, badge_y2), color, -1)
                        cv2.putText(
                            frame, label_text, (bx1 + 5, badge_y2 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                        )

                writer.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            writer.release()

        # Transcode to H.264 MP4 for browser playback; fall back to the raw
        # cv2 output if ffmpeg is somehow unavailable (job still succeeds).
        try:
            self._transcode_to_h264(writer_path, play_path)
        except Exception as tr_err:
            logger.warning(f"H.264 transcode failed ({tr_err}); serving cv2 encoder output instead.")
            shutil.move(writer_path, play_path)
        else:
            os.remove(writer_path)

        logger.info(f"Rendered annotated video successfully to: '{play_path}'")
        return play_path

    @staticmethod
    def _transcode_to_h264(src_path: str, dst_path: str) -> None:
        """Transcodes the cv2-written video to H.264 MP4 using bundled ffmpeg."""
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-loglevel", "error",
            "-i", src_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "26",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an", dst_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg transcode failed: {result.stderr[:400]}")
        if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
            raise RuntimeError("ffmpeg produced no output")
