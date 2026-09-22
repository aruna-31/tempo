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


class TiledYOLOPersonDetector(YOLOPersonDetector):
    """Opt-in tiled person detector for dense, small-person classroom views."""

    def __init__(
        self,
        weights_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.15,
        image_size: int = 1280,
        iou_threshold: float = 0.55,
        tile_size: int = 640,
        tile_overlap: float = 0.25,
        merge_iou: float = 0.55,
        context_margin: float = 0.12,
    ):
        super().__init__(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            context_margin=context_margin,
        )
        self.image_size = int(image_size)
        self.iou_threshold = float(iou_threshold)
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.merge_iou = float(merge_iou)
        self.last_raw_detection_count = 0
        self.last_raw_detections: List[Dict[str, Any]] = []

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    @classmethod
    def _containment(cls, a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        smaller = min(
            max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]),
            max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]),
        )
        return intersection / smaller if smaller else 0.0

    def _merge_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        remaining = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        merged: List[Dict[str, Any]] = []
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            keep = []
            for candidate in remaining:
                if (
                    self._iou(seed["bbox"], candidate["bbox"]) >= self.merge_iou
                    or self._containment(seed["bbox"], candidate["bbox"]) >= 0.75
                ):
                    group.append(candidate)
                else:
                    keep.append(candidate)
            remaining = keep
            weights = [max(0.01, item["confidence"]) for item in group]
            total_weight = sum(weights)
            bbox = [
                sum(item["bbox"][axis] * weights[index] for index, item in enumerate(group)) / total_weight
                for axis in range(4)
            ]
            best = max(group, key=lambda item: item["confidence"])
            merged.append({**best, "bbox": [round(value, 2) for value in bbox]})
        return merged

    def detect_persons(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty frame supplied for tiled detection")
        height, width = frame.shape[:2]
        stride = max(1, int(self.tile_size * (1.0 - self.tile_overlap)))
        x_starts = list(range(0, max(1, width - self.tile_size + 1), stride))
        y_starts = list(range(0, max(1, height - self.tile_size + 1), stride))
        if not x_starts or x_starts[-1] != max(0, width - self.tile_size):
            x_starts.append(max(0, width - self.tile_size))
        if not y_starts or y_starts[-1] != max(0, height - self.tile_size):
            y_starts.append(max(0, height - self.tile_size))

        raw: List[Dict[str, Any]] = []
        for top in sorted(set(y_starts)):
            for left in sorted(set(x_starts)):
                tile = frame[top:min(height, top + self.tile_size), left:min(width, left + self.tile_size)]
                result = self.model.predict(
                    source=tile,
                    classes=[0],
                    conf=self.confidence_threshold,
                    imgsz=self.image_size,
                    iou=self.iou_threshold,
                    verbose=False,
                )[0]
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if int(box.cls[0].cpu().numpy()) != 0:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    bbox = [
                        max(0.0, min(float(width), float(xyxy[0] + left))),
                        max(0.0, min(float(height), float(xyxy[1] + top))),
                        max(0.0, min(float(width), float(xyxy[2] + left))),
                        max(0.0, min(float(height), float(xyxy[3] + top))),
                    ]
                    if bbox[2] - bbox[0] >= 5 and bbox[3] - bbox[1] >= 5:
                        raw.append({
                            "bbox": bbox,
                            "confidence": float(box.conf[0].cpu().numpy()),
                            "class_name": "person",
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                        })
        self.last_raw_detection_count = len(raw)
        self.last_raw_detections = raw
        return self._merge_detections(raw)


class HighRecallTiledDetector(TiledYOLOPersonDetector):
    """High-recall detector optimised for dense classroom person localization.

    Provides high-resolution native inference (imgsz=1280) with tuned NMS IoU (0.45)
    and confidence floor (0.22) that eliminates tile boundary seam cuts (which
    previously split students and caused track fragmentation).

    Optionally supports multi-scale fine tiling (``use_tiles=True``) for extra-dense
    scenes, with robust duplicate suppression (containment threshold 0.85, aspect ratio
    filtering 0.17 - 2.5).

    This class is opt-in (``CLASSROOM_DETECTOR_MODE=highrecall``) and does
    **not** modify ``TiledYOLOPersonDetector`` or any production defaults.
    """

    def __init__(
        self,
        weights_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.23,
        image_size: int = 1280,
        iou_threshold: float = 0.40,
        # coarse-tile settings (inherited)
        tile_size: int = 640,
        tile_overlap: float = 0.25,
        # fine-tile settings
        fine_tile_size: int = 480,
        fine_overlap: float = 0.40,
        # improved merge / suppression
        merge_iou: float = 0.40,
        containment_threshold: float = 0.85,
        min_aspect_ratio: float = 0.17,   # reject width/height < 0.17
        max_aspect_ratio: float = 2.5,    # reject width/height > 2.5
        context_margin: float = 0.12,
        use_tiles: bool = False,
    ):
        super().__init__(
            weights_path=weights_path,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            iou_threshold=iou_threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            merge_iou=merge_iou,
            context_margin=context_margin,
        )
        self.fine_tile_size = int(fine_tile_size)
        self.fine_overlap = float(fine_overlap)
        self.containment_threshold = float(containment_threshold)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.use_tiles = bool(use_tiles)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_full_frame(
        self, frame: np.ndarray, frame_idx: int, timestamp: float,
    ) -> List[Dict[str, Any]]:
        """Single full-frame YOLO pass at ``image_size`` resolution."""
        height, width = frame.shape[:2]
        result = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            iou=self.iou_threshold,
            verbose=False,
        )[0]
        detections: List[Dict[str, Any]] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            if int(box.cls[0].cpu().numpy()) != 0:
                continue
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            bbox = [
                max(0.0, min(float(width), float(xyxy[0]))),
                max(0.0, min(float(height), float(xyxy[1]))),
                max(0.0, min(float(width), float(xyxy[2]))),
                max(0.0, min(float(height), float(xyxy[3]))),
            ]
            if bbox[2] - bbox[0] >= 5 and bbox[3] - bbox[1] >= 5:
                detections.append({
                    "bbox": bbox,
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "class_name": "person",
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                })
        return detections

    def _run_tiles(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        tile_sz: int,
        overlap: float,
    ) -> List[Dict[str, Any]]:
        """Run tiled inference with the given tile size and overlap."""
        height, width = frame.shape[:2]
        stride = max(1, int(tile_sz * (1.0 - overlap)))
        x_starts = list(range(0, max(1, width - tile_sz + 1), stride))
        y_starts = list(range(0, max(1, height - tile_sz + 1), stride))
        if not x_starts or x_starts[-1] != max(0, width - tile_sz):
            x_starts.append(max(0, width - tile_sz))
        if not y_starts or y_starts[-1] != max(0, height - tile_sz):
            y_starts.append(max(0, height - tile_sz))

        raw: List[Dict[str, Any]] = []
        for top in sorted(set(y_starts)):
            for left in sorted(set(x_starts)):
                tile = frame[
                    top : min(height, top + tile_sz),
                    left : min(width, left + tile_sz),
                ]
                result = self.model.predict(
                    source=tile,
                    classes=[0],
                    conf=self.confidence_threshold,
                    imgsz=self.image_size,
                    iou=self.iou_threshold,
                    verbose=False,
                )[0]
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if int(box.cls[0].cpu().numpy()) != 0:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    bbox = [
                        max(0.0, min(float(width), float(xyxy[0] + left))),
                        max(0.0, min(float(height), float(xyxy[1] + top))),
                        max(0.0, min(float(width), float(xyxy[2] + left))),
                        max(0.0, min(float(height), float(xyxy[3] + top))),
                    ]
                    if bbox[2] - bbox[0] >= 5 and bbox[3] - bbox[1] >= 5:
                        raw.append({
                            "bbox": bbox,
                            "confidence": float(box.conf[0].cpu().numpy()),
                            "class_name": "person",
                            "frame_idx": frame_idx,
                            "timestamp": timestamp,
                        })
        return raw

    def _filter_aspect_ratio(
        self, detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reject boxes with extreme aspect ratios (desk/furniture fragments)."""
        kept: List[Dict[str, Any]] = []
        for det in detections:
            b = det["bbox"]
            w = max(1e-3, b[2] - b[0])
            h = max(1e-3, b[3] - b[1])
            ratio = w / h
            if self.min_aspect_ratio <= ratio <= self.max_aspect_ratio:
                kept.append(det)
        return kept

    def _merge_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter aspect ratio and merge overlapping detections without corrupting nearby students."""
        detections = self._filter_aspect_ratio(detections)
        if not self.use_tiles and len(detections) <= 1:
            return detections
        remaining = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        merged: List[Dict[str, Any]] = []
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            keep = []
            for candidate in remaining:
                iou_val = self._iou(seed["bbox"], candidate["bbox"])
                cont_val = self._containment(seed["bbox"], candidate["bbox"])
                if iou_val >= self.merge_iou or cont_val >= self.containment_threshold:
                    group.append(candidate)
                else:
                    keep.append(candidate)
            remaining = keep
            weights = [max(0.01, d["confidence"]) for d in group]
            total_w = sum(weights)
            bbox = [
                sum(d["bbox"][ax] * weights[i] for i, d in enumerate(group)) / total_w
                for ax in range(4)
            ]
            best = max(group, key=lambda d: d["confidence"])
            merged.append({**best, "bbox": [round(v, 2) for v in bbox]})
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_persons(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty frame for high-recall detection")

        # Pass 1: High-resolution full frame at native aspect ratio
        full_dets = self._run_full_frame(frame, frame_idx, timestamp)

        if self.use_tiles:
            # Pass 2: Fine tiles for dense scenes if explicitly enabled
            fine_dets = self._run_tiles(
                frame, frame_idx, timestamp,
                tile_sz=self.fine_tile_size,
                overlap=self.fine_overlap,
            )
            raw = full_dets + fine_dets
        else:
            raw = full_dets

        self.last_raw_detection_count = len(raw)
        self.last_raw_detections = list(raw)
        return self._merge_detections(raw)
