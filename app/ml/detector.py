import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("tempo.ml.detector")


class YOLOPersonDetector:
    """
    Real Ultralytics YOLO Person Detector service.
    Detects students/persons in classroom video frames and extracts
    appropriately sized person crops with spatial context and aspect-ratio preservation.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.30,
        context_margin: float = 0.12  # 12% margin around bbox to preserve posture and desk context
    ):
        self.confidence_threshold = confidence_threshold
        self.context_margin = max(0.0, min(0.5, context_margin))
        self.weights_path = weights_path or "yolov8n.pt"

        if weights_path and not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights file not found at '{weights_path}'")

        try:
            logger.info(f"Loading Ultralytics YOLO model from '{self.weights_path}'...")
            self.model = YOLO(self.weights_path)
            logger.info("Successfully loaded Ultralytics YOLO person detector.")
        except Exception as e:
            logger.error(f"Failed to load Ultralytics YOLO model: {e}")
            raise RuntimeError(f"YOLO initialization failed: {e}") from e

    def detect_persons(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Detects person bounding boxes in a single frame using Ultralytics YOLO.
        Only person class (COCO class 0) is returned.
        Raises ValueError if frame is invalid or empty.
        Returns:
            List of detections: [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_name": "person",
                    "frame_idx": int,
                    "timestamp": float
                }
            ]
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError(f"Invalid or empty frame supplied for detection at frame {frame_idx}")

        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            raise ValueError(f"Frame dimensions too small ({w}x{h}) for detection at frame {frame_idx}")

        detections: List[Dict[str, Any]] = []

        try:
            # Predict class 0 (person) at a lowered floor so ByteTrack's second
            # association receives the low-score pool it needs to keep tracks
            # alive through occlusion; scores below track_thresh are filtered
            # by the tracker, not lost to the detector's cut.
            results = self.model.predict(
                source=frame,
                classes=[0],
                conf=min(self.confidence_threshold, 0.10),
                verbose=False
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    if cls_id == 0:  # Person class
                        x1 = max(0.0, min(float(w), float(xyxy[0])))
                        y1 = max(0.0, min(float(h), float(xyxy[1])))
                        x2 = max(0.0, min(float(w), float(xyxy[2])))
                        y2 = max(0.0, min(float(h), float(xyxy[3])))

                        if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                            detections.append({
                                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                                "confidence": round(conf, 4),
                                "class_name": "person",
                                "frame_idx": frame_idx,
                                "timestamp": timestamp
                            })

        except Exception as e:
            logger.error(f"Error during YOLO person detection on frame {frame_idx}: {e}")
            raise RuntimeError(f"YOLO person detection error on frame {frame_idx}: {e}") from e

        return detections

    def extract_person_crop(
        self,
        frame: np.ndarray,
        bbox: List[float],
        target_size: Optional[Tuple[int, int]] = (224, 224)
    ) -> np.ndarray:
        """
        Extracts a person crop with spatial context margin and aspect-ratio preservation,
        avoiding anamorphic distortion or excessive zooming.
        Args:
            frame: (H, W, 3) image
            bbox: [x1, y1, x2, y2]
            target_size: Optional (width, height) to resize crop to with letterbox padding
        Returns:
            np.ndarray crop with target_size
        """
        if frame is None or frame.size == 0:
            if target_size:
                return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            return np.zeros((224, 224, 3), dtype=np.uint8)

        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        # Apply proportional context margin
        margin_w = bw * self.context_margin
        margin_h = bh * self.context_margin

        crop_x1 = max(0, int(round(x1 - margin_w)))
        crop_y1 = max(0, int(round(y1 - margin_h)))
        crop_x2 = min(w_img, int(round(x2 + margin_w)))
        crop_y2 = min(h_img, int(round(y2 + margin_h)))

        # Ensure valid non-empty slice
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raw_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            raw_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if target_size is None:
            return raw_crop

        target_w, target_h = target_size
        ch, cw = raw_crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Scale proportionally to fit inside target_size while preserving aspect ratio
        scale = min(target_w / cw, target_h / ch)
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))

        resized = cv2.resize(raw_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Pad symmetrically onto canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        return canvas
