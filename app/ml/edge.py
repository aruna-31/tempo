"""
TEMPO — Edge Processing Subsystem
=================================
Implements the Edge Layer perception abstraction:
  1. Video/Frame Acquisition: samples decodable frames at configured sampling rate.
  2. High-Resolution & Adaptive ROI Detection: detects student persons with duplicate suppression.
  3. Anonymous Multi-Object Tracking: assigns and maintains session-scoped anonymous track IDs.
  4. Context-Preserved Student Crop Extraction: extracts 224x224 person crops.
  5. ResNet-50 + BiGRU + Temporal Attention Inference: extracts spatial features and predicts behaviour.
  6. Anonymous Temporal Event Emission: emits structured events and track trajectories
     WITHOUT transferring raw video frames to the central server.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.ml.appearance import AnonymousAppearanceEmbedder
from app.ml.detector import (
    AdaptiveHighResObservationDetector,
    HighRecallTiledDetector,
    TiledYOLOPersonDetector,
    YOLOPersonDetector,
)
from app.ml.model import (
    CLASS_NAMES,
    ResNet18TemporalModel,
    ResNet50BiGRUTemporalModel,
    load_tempo_behaviour_model,
)
from app.ml.observation import is_back_row_bbox
from app.ml.preprocessor import preprocessor
from app.ml.sampler import TrackSequenceBuilder, VideoFrameSampler
from app.ml.tracker import ByteTracker, DenseByteTracker

logger = logging.getLogger("tempo.ml.edge")


@dataclass
class AnonymousTemporalEvent:
    """Represents a single anonymized student behaviour event emitted by EdgeProcessing."""
    track_id: int
    student_label: str
    frame_number: int
    timestamp_seconds: float
    window_start_sec: float
    window_end_sec: float
    bbox: List[float]
    detection_confidence: float
    behaviour_type: str
    confidence: float
    probability_distribution: Dict[str, float]
    attention_weights: Optional[List[float]] = None
    is_back_row: bool = False
    model: str = "ResNet50_BiGRU"
    model_version: str = "v3.0.0-resnet50-bigru-attention"

    @property
    def anonymous_student_id(self) -> str:
        return self.student_label

    @property
    def start_time(self) -> float:
        return self.window_start_sec

    @property
    def end_time(self) -> float:
        return self.window_end_sec

    @property
    def duration(self) -> float:
        return round(self.window_end_sec - self.window_start_sec, 2)

    @property
    def behavior(self) -> str:
        return self.behaviour_type

    @property
    def bounding_box(self) -> List[float]:
        return self.bbox

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeProcessingOutput:
    """Aggregated output payload emitted by EdgeProcessing for room/fog consumption."""
    session_id: str
    events: List[AnonymousTemporalEvent] = field(default_factory=list)
    track_trajectories: List[Dict[str, Any]] = field(default_factory=list)
    total_sampled_frames: int = 0
    video_duration_seconds: float = 0.0
    edge_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "events": [e.to_dict() for e in self.events],
            "track_trajectories": self.track_trajectories,
            "total_sampled_frames": self.total_sampled_frames,
            "video_duration_seconds": self.video_duration_seconds,
            "edge_metadata": self.edge_metadata,
        }


class EdgeProcessing:
    """
    Edge Node Processing Abstraction.

    Handles high-speed video perception locally at the camera/edge:
      - Raw frame decodability & acquisition.
      - Adaptive high-resolution detection on small/back-row students.
      - Anonymous session-specific tracking (zero face recognition, zero permanent IDs).
      - Multi-student temporal sequence building.
      - ResNet-50 + 2-layer BiGRU + Temporal Attention forward inference.
      - Output: Anonymous temporal events emitted to Fog layer without raw video transfer.
    """

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        model_weights_path: Optional[str] = None,
        device_name: Optional[str] = None,
        detector_mode: Optional[str] = None,
        context_margin: float = 0.12,
    ):
        self.weights_path = model_weights_path or settings.MODEL_WEIGHTS_PATH
        if not os.path.isabs(self.weights_path):
            self.weights_path = os.path.abspath(self.weights_path)
        self.context_margin = context_margin

        if metadata_path is not None:
            self.metadata_path = metadata_path
        else:
            meta_candidate = os.path.join(os.path.dirname(self.weights_path), "model_metadata.json")
            self.metadata_path = os.path.abspath(meta_candidate)

        # 1. Load metadata as single source of truth
        self.metadata = self._load_metadata(self.metadata_path)
        self.model_version = str(self.metadata["model_version"])
        self.spatial_backbone = str(self.metadata["spatial_backbone"]).lower()
        self.temporal_type = str(self.metadata["temporal_model_type"]).upper()
        self.hidden_dim = int(self.metadata["hidden_dim"])
        self.num_layers = int(self.metadata["num_layers"])
        self.sequence_length = int(self.metadata["sequence_length"])
        self.sampling_fps = float(self.metadata["sampling_fps"])
        self.classes: List[str] = list(self.metadata["classes"])

        # 2. Device Selection
        req_dev = device_name or settings.ML_DEVICE
        if req_dev == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(req_dev if req_dev in ("cuda", "cpu") else "cpu")

        # 3. Model Initialization & Strict Weights Loading
        self.model = load_tempo_behaviour_model(self.metadata, weights_path=self.weights_path)
        self.model.to(self.device)
        self.model.eval()

        # 4. Sampler & Sequence Builder
        self.preprocessor = preprocessor
        self.sampler = VideoFrameSampler(
            sampling_fps=self.sampling_fps,
            sequence_length=self.sequence_length,
            stride=max(1, self.sequence_length // 2),
        )
        self.sequence_builder = TrackSequenceBuilder(
            sequence_length=self.sequence_length,
            stride=max(1, self.sequence_length // 2),
        )

        # 5. Detector Initialization
        det_mode = (detector_mode or settings.CLASSROOM_DETECTOR_MODE).lower()
        self.detector_mode = det_mode
        dense_weights = getattr(settings, "CLASSROOM_DETECTOR_WEIGHTS", "./yolov8s.pt")
        if not os.path.isabs(dense_weights):
            dense_weights = os.path.abspath(dense_weights)

        if det_mode in ("adaptive", "adaptive_highres"):
            self.detector = AdaptiveHighResObservationDetector(
                weights_path=dense_weights,
                confidence_threshold=getattr(settings, "CLASSROOM_ADAPTIVE_CONFIDENCE_BASE", 0.23),
                adaptive_confidence=getattr(settings, "CLASSROOM_ADAPTIVE_CONFIDENCE", 0.18),
                image_size=getattr(settings, "CLASSROOM_DETECTOR_IMAGE_SIZE", 1280),
                iou_threshold=getattr(settings, "CLASSROOM_DETECTOR_IOU", 0.40),
                back_row_split=getattr(settings, "CLASSROOM_ADAPTIVE_ROI_SPLIT", 0.50),
                tile_size=getattr(settings, "CLASSROOM_ADAPTIVE_TILE_SIZE", 540),
                tile_overlap=getattr(settings, "CLASSROOM_ADAPTIVE_TILE_OVERLAP", 0.35),
                merge_iou=getattr(settings, "CLASSROOM_ADAPTIVE_MERGE_IOU", 0.40),
                containment_threshold=getattr(settings, "CLASSROOM_ADAPTIVE_CONTAINMENT", 0.80),
                context_margin=self.context_margin,
            )
        elif det_mode in ("tiled", "highrecall"):
            if det_mode == "highrecall":
                self.detector = HighRecallTiledDetector(
                    weights_path=dense_weights,
                    confidence_threshold=getattr(settings, "CLASSROOM_HIGHRECALL_CONFIDENCE", 0.22),
                    image_size=settings.CLASSROOM_DETECTOR_IMAGE_SIZE,
                    iou_threshold=getattr(settings, "CLASSROOM_HIGHRECALL_MERGE_IOU", 0.45),
                    tile_size=settings.CLASSROOM_TILE_SIZE,
                    tile_overlap=settings.CLASSROOM_TILE_OVERLAP,
                    fine_tile_size=settings.CLASSROOM_HIGHRECALL_FINE_TILE,
                    fine_overlap=settings.CLASSROOM_HIGHRECALL_FINE_OVERLAP,
                    merge_iou=settings.CLASSROOM_HIGHRECALL_MERGE_IOU,
                    containment_threshold=settings.CLASSROOM_HIGHRECALL_CONTAINMENT,
                    context_margin=self.context_margin,
                    use_tiles=getattr(settings, "CLASSROOM_HIGHRECALL_USE_TILES", False),
                )
            else:
                self.detector = TiledYOLOPersonDetector(
                    weights_path=dense_weights,
                    confidence_threshold=settings.CLASSROOM_DETECTOR_CONFIDENCE,
                    image_size=settings.CLASSROOM_DETECTOR_IMAGE_SIZE,
                    iou_threshold=settings.CLASSROOM_DETECTOR_IOU,
                    tile_size=settings.CLASSROOM_TILE_SIZE,
                    tile_overlap=settings.CLASSROOM_TILE_OVERLAP,
                    merge_iou=settings.CLASSROOM_NMS_IOU,
                    context_margin=self.context_margin,
                )
        else:
            self.detector = YOLOPersonDetector(
                weights_path=os.path.abspath("yolov8n.pt"),
                confidence_threshold=0.30,
                context_margin=self.context_margin,
            )

        # 6. Anonymous Tracker
        if settings.CLASSROOM_TRACKER_MODE.lower() == "dense":
            self.tracker = DenseByteTracker(
                track_thresh=0.30,
                match_thresh=0.80,
                match_thresh_second=0.40,
                max_time_lost_frames=settings.CLASSROOM_TRACKER_LOST_BUFFER,
                confirmation_hits=settings.CLASSROOM_TRACKER_CONFIRMATION_HITS,
                center_distance_gate=settings.CLASSROOM_TRACKER_CENTER_GATE,
                appearance_weight=settings.CLASSROOM_TRACKER_APPEARANCE_WEIGHT,
                appearance_gate=settings.CLASSROOM_TRACKER_APPEARANCE_GATE,
            )
            self.appearance_embedder = AnonymousAppearanceEmbedder()
        else:
            self.tracker = ByteTracker(
                track_thresh=0.30,
                match_thresh=0.80,
                match_thresh_second=0.40,
                max_time_lost_frames=90,
            )
            self.appearance_embedder = None

        logger.info(
            f"Initialized EdgeProcessing: Backbone={self.spatial_backbone}, "
            f"Temporal={self.temporal_type}, DetectorMode={self.detector_mode}, Device={self.device}"
        )

    def _load_metadata(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model metadata not found at '{path}'")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _consolidate_track_fragments(
        self,
        track_results: List[Dict[str, Any]],
        max_gap_frames: int = 300,
        iou_threshold: float = 0.20,
        min_track_observations: int = 3,
    ) -> Dict[int, int]:
        """Consolidates tracker ID fragments into seat-level student tracks."""
        def box_iou(a: List[float], b: List[float]) -> float:
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / union if union > 0 else 0.0

        tracks = [t for t in track_results if t.get("bounding_box_history")]
        if len(tracks) <= 1:
            return {}

        def span(t):
            h = t["bounding_box_history"]
            return h[0]["frame"], h[-1]["frame"]

        by_id = {t["track_id"]: t for t in tracks}
        ids = [t["track_id"] for t in tracks]
        obs = {t["track_id"]: len(t["bounding_box_history"]) for t in tracks}

        parent = {i: i for i in ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        histories = {t["track_id"]: t["bounding_box_history"] for t in tracks}
        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                a, b = by_id[ids[ii]], by_id[ids[jj]]
                a_start, a_end = span(a)
                b_start, b_end = span(b)
                gap = max(a_start - b_end, b_start - a_end, 0)
                if gap > max_gap_frames:
                    continue
                ha, hb = histories[a["track_id"]], histories[b["track_id"]]
                if gap == 0:
                    fa = {h["frame"]: h["bbox"] for h in ha}
                    shared = [box_iou(fa[f], h["bbox"]) for h in hb if (f := h["frame"]) in fa]
                    if shared and float(np.mean(shared)) >= iou_threshold:
                        union(a["track_id"], b["track_id"])
                else:
                    first_b = hb[0]["bbox"]
                    last_a = ha[-1]["bbox"]
                    if box_iou(first_b, last_a) >= iou_threshold:
                        union(a["track_id"], b["track_id"])

        groups = defaultdict(list)
        for i in ids:
            groups[find(i)].append(i)

        assignment: Dict[int, int] = {}
        for root, members in groups.items():
            members.sort(key=lambda m: (-obs[m], m))
            total = sum(obs[m] for m in members)
            if obs[members[0]] >= 0.7 * total:
                keep = members[0]
                for m in members:
                    assignment[m] = keep
            else:
                for m in members:
                    assignment[m] = m
        return assignment

    def _filter_fragment_tracks(
        self,
        merged_histories: Dict[int, List[Dict[str, Any]]],
        min_distinct_frames: int = 10,
        presence_ratio: float = 0.42,
    ) -> Dict[int, List[Dict[str, Any]]]:
        if not merged_histories:
            return {}
        distinct = {
            cid: len({h["frame"] for h in hs})
            for cid, hs in merged_histories.items()
        }
        threshold = max(min_distinct_frames, round(presence_ratio * max(distinct.values())))
        return {
            cid: hs for cid, hs in merged_histories.items()
            if distinct[cid] >= threshold
        }

    def process_camera_source(
        self,
        camera_source: Any,
        session_id: Optional[str] = None,
        max_duration_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> EdgeProcessingOutput:
        """
        Executes edge pipeline on frames extracted from a BaseCameraSource
        (supports DemoMP4CameraSource, RTSPStreamCameraSource, etc.).
        """
        session_uuid = session_id or str(uuid.uuid4())
        source_name = getattr(camera_source, "name", "Camera")
        source_type = getattr(camera_source, "source_type", "DEMO")
        logger.info(f"[Edge] Ingesting from camera source: '{source_name}' ({source_type}), Session: {session_uuid}")
        if progress_callback:
            progress_callback(5)

        self.tracker.reset()

        sampled_frames = camera_source.extract_frames(
            sampling_fps=self.sampling_fps,
            max_duration_seconds=max_duration_seconds,
        )
        if not sampled_frames:
            raise ValueError(f"No decodable frames received from camera source '{source_name}'")

        ref = camera_source.get_source_reference() if hasattr(camera_source, "get_source_reference") else source_name
        return self._process_sampled_frames(
            sampled_frames=sampled_frames,
            session_uuid=session_uuid,
            source_ref=ref,
            progress_callback=progress_callback,
        )

    def process_video(
        self,
        video_path: str,
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> EdgeProcessingOutput:
        """
        Executes end-to-end edge pipeline:
          Acquisition -> Adaptive Detection -> Anonymous Tracking -> ResNet-50 BiGRU Inference.
        Returns:
          EdgeProcessingOutput containing anonymous temporal events and track trajectories.
        """
        session_uuid = session_id or str(uuid.uuid4())
        logger.info(f"[Edge] Processing video: '{video_path}', Session: {session_uuid}")
        if progress_callback:
            progress_callback(5)

        self.tracker.reset()

        # 1. Acquire sampled frames from video
        sampled_frames = self.sampler.extract_frames(video_path)
        if not sampled_frames:
            raise ValueError(f"No decodable frames extracted from video '{video_path}'")

        return self._process_sampled_frames(
            sampled_frames=sampled_frames,
            session_uuid=session_uuid,
            source_ref=video_path,
            progress_callback=progress_callback,
        )

    def _process_sampled_frames(
        self,
        sampled_frames: List[Dict[str, Any]],
        session_uuid: str,
        source_ref: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> EdgeProcessingOutput:
        total_frames = len(sampled_frames)
        duration_sec = total_frames / max(0.1, self.sampling_fps)

        if progress_callback:
            progress_callback(20)

        # 2. Adaptive Detection and Anonymous Tracking
        track_crops: Dict[int, List[Dict[str, Any]]] = {}

        for idx, item in enumerate(sampled_frames):
            frame = item["frame_data"]
            f_num = item["frame_number"]
            t_sec = item["timestamp_seconds"]

            detections = self.detector.detect_persons(frame, frame_idx=f_num, timestamp=t_sec)

            if self.appearance_embedder is not None:
                for det in detections:
                    det["appearance"] = self.appearance_embedder(frame, det["bbox"])

            active_stracks = self.tracker.update(detections, frame_idx=f_num, timestamp_seconds=t_sec)

            for strack in active_stracks:
                t_id = strack.track_id
                if t_id not in track_crops:
                    track_crops[t_id] = []

                crop = self.detector.extract_person_crop(frame, strack.bbox.tolist(), target_size=(224, 224))
                track_crops[t_id].append({
                    "frame_number": f_num,
                    "timestamp_seconds": t_sec,
                    "crop": crop,
                    "bbox": [round(float(c), 2) for c in strack.bbox],
                    "confidence": float(strack.score),
                })

            if progress_callback:
                pct = 20 + int((idx + 1) / total_frames * 25)
                progress_callback(min(45, pct))

        # Consolidate tracker fragments
        all_session_tracks = self.tracker.get_all_session_tracks()
        raw_tracks = [
            {
                "track_id": s.track_id,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "bounding_box_history": s.history,
            }
            for s in all_session_tracks
        ]

        consolidation = self._consolidate_track_fragments(raw_tracks)
        merged_histories: Dict[int, List[Dict[str, Any]]] = {}
        for t in raw_tracks:
            cid = consolidation.get(t["track_id"], t["track_id"])
            merged_histories.setdefault(cid, []).extend(t["bounding_box_history"])

        rekeyed_crops: Dict[int, List[Dict[str, Any]]] = {}
        for tid, obs in track_crops.items():
            rekeyed_crops.setdefault(consolidation.get(tid, tid), []).extend(obs)

        min_frames_req = min(10, max(1, total_frames // 3))
        merged_histories = self._filter_fragment_tracks(merged_histories, min_distinct_frames=min_frames_req)

        first_frame = {cid: min(h["frame"] for h in hs) for cid, hs in merged_histories.items()}
        renumber = {
            cid: new_id
            for new_id, cid in enumerate(sorted(first_frame, key=lambda c: first_frame[c]), start=1)
        }

        final_track_trajectories = [
            {
                "track_id": renumber[cid],
                "start_frame": min(h["frame"] for h in hs),
                "end_frame": max(h["frame"] for h in hs),
                "bounding_box_history": sorted(hs, key=lambda h: h["frame"]),
            }
            for cid, hs in merged_histories.items()
        ]
        final_track_trajectories.sort(key=lambda t: t["track_id"])
        final_track_crops = {renumber[cid]: obs for cid, obs in rekeyed_crops.items() if cid in renumber}

        if progress_callback:
            progress_callback(50)

        # 3. Build per-track temporal sequences
        track_sequences = self.sequence_builder.build_track_sequences(final_track_crops)
        if not track_sequences:
            logger.warning(f"[Edge] Zero student tracks found in '{video_path}'.")
            return EdgeProcessingOutput(
                session_id=session_uuid,
                events=[],
                track_trajectories=final_track_trajectories,
                total_sampled_frames=total_frames,
                video_duration_seconds=duration_sec,
                edge_metadata={
                    "model_version": self.model_version,
                    "detector_mode": self.detector_mode,
                    "spatial_backbone": self.spatial_backbone,
                },
            )

        # 4. ResNet-50 + BiGRU + Temporal Attention Inference
        emitted_events: List[AnonymousTemporalEvent] = []
        total_tracks = len(track_sequences)
        processed = 0
        frame_h = sampled_frames[0]["frame_data"].shape[0] if sampled_frames else 480

        model_label = f"{self.spatial_backbone.capitalize()}_{self.temporal_type}"

        with torch.no_grad():
            for track_id, sequences in sorted(track_sequences.items(), key=lambda x: x[0]):
                student_label = f"Student {track_id:02d}"
                batch_size = 4
                num_seqs = len(sequences)

                for i in range(0, num_seqs, batch_size):
                    batch_seqs = sequences[i:i + batch_size]
                    batch_tensors = [self.preprocessor.preprocess_batch(s["crops"]) for s in batch_seqs]
                    batch_input = torch.stack(batch_tensors, dim=0).to(self.device)

                    if isinstance(self.model, ResNet50BiGRUTemporalModel):
                        logits, attn_weights = self.model(batch_input, return_attention=True)
                        attn_np = attn_weights.cpu().numpy() if attn_weights is not None else None
                    else:
                        logits = self.model(batch_input)
                        attn_np = None

                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    top_indices = torch.argmax(logits, dim=-1).cpu().numpy()

                    for j, s in enumerate(batch_seqs):
                        pred_idx = int(top_indices[j])
                        pred_label = self.classes[pred_idx]
                        conf = float(probs[j][pred_idx])

                        prob_dist = {
                            self.classes[k]: round(float(probs[j][k]), 4)
                            for k in range(len(self.classes))
                        }
                        cur_attn = (
                            [round(float(a), 4) for a in attn_np[j]]
                            if attn_np is not None
                            else None
                        )

                        # Find bounding box at anchor frame
                        anchor_box = [0.0, 0.0, 0.0, 0.0]
                        anchor_det_conf = 0.80
                        for c_item in final_track_crops.get(track_id, []):
                            if c_item["frame_number"] == s["anchor_frame_number"]:
                                anchor_box = c_item["bbox"]
                                anchor_det_conf = c_item.get("confidence", 0.80)
                                break

                        is_br = is_back_row_bbox(anchor_box, frame_height=frame_h)

                        event = AnonymousTemporalEvent(
                            track_id=track_id,
                            student_label=student_label,
                            frame_number=s["anchor_frame_number"],
                            timestamp_seconds=round(s["anchor_timestamp"], 3),
                            window_start_sec=round(s["window_start_time"], 3),
                            window_end_sec=round(s["window_end_time"], 3),
                            bbox=anchor_box,
                            detection_confidence=round(anchor_det_conf, 4),
                            behaviour_type=pred_label,
                            confidence=round(conf, 4),
                            probability_distribution=prob_dist,
                            attention_weights=cur_attn,
                            is_back_row=is_br,
                            model=model_label,
                            model_version=self.model_version,
                        )
                        emitted_events.append(event)

                processed += 1
                if progress_callback:
                    pct = 50 + int((processed / total_tracks) * 35)
                    progress_callback(min(85, pct))

        logger.info(
            f"[Edge] Emitted {len(emitted_events)} anonymous temporal events "
            f"for {len(final_track_trajectories)} student tracks (Session: {session_uuid})"
        )

        return EdgeProcessingOutput(
            session_id=session_uuid,
            events=emitted_events,
            track_trajectories=final_track_trajectories,
            total_sampled_frames=total_frames,
            video_duration_seconds=round(duration_sec, 2),
            edge_metadata={
                "model_version": self.model_version,
                "spatial_backbone": self.spatial_backbone,
                "temporal_model_type": self.temporal_type,
                "detector_mode": self.detector_mode,
                "device": str(self.device),
            },
        )
