import json
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from app.core.config import settings
from app.ml.annotator import VideoAnnotator
from app.ml.appearance import AnonymousAppearanceEmbedder
from app.ml.detector import HighRecallTiledDetector, TiledYOLOPersonDetector, YOLOPersonDetector
from app.ml.model import ResNet18TemporalModel
from app.ml.preprocessor import preprocessor
from app.ml.sampler import TrackSequenceBuilder, VideoFrameSampler
from app.ml.tracker import ByteTracker, DenseByteTracker

logger = logging.getLogger("tempo.ml.service")

_REQUIRED_METADATA_KEYS = (
    "model_version",
    "spatial_backbone",
    "temporal_model_type",
    "hidden_dim",
    "num_layers",
    "sequence_length",
    "sampling_fps",
    "input_size",
    "classes",
    "class_mapping",
    "normalization",
)


class MLInferenceService:
    """
    High-level Individual Student-Track Temporal Behaviour Inference Service.
    Orchestrates:
      1. Configuration from model_metadata.json as the SINGLE SOURCE OF TRUTH (strict, no defaults)
      2. Strict Checkpoint Loading (ResNet-18 + Temporal RNN/LSTM/GRU) - architecture must exactly match metadata
      3. Frame sampling & Temporal Windowing (VideoFrameSampler)
      4. Person Detection (Ultralytics YOLO)
      5. Multi-Object Tracking & Session Track ID Maintenance (ByteTracker)
      6. Context & Aspect-Ratio Preserving Person Crop Extraction
      7. Per-Track Independent Temporal Sequence Building (TrackSequenceBuilder)
      8. Spatial ResNet-18 (512-dim) + Temporal Model Inference (5 Observable Classes)
      9. Structured Result Aggregation and Annotated Output Video Rendering (VideoAnnotator)

    Never silently uses random weights, synthetic predictions, fallback detections or fallback frames.
    Invalid weights, missing model files, missing YOLO weights or undecodable video raise exceptions
    which fail the AnalysisJob with the actual error message.
    """

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        model_weights_path: Optional[str] = None,
        device_name: Optional[str] = None,
        context_margin: float = 0.12
    ):
        self.weights_path = model_weights_path or settings.MODEL_WEIGHTS_PATH
        if not os.path.isabs(self.weights_path):
            self.weights_path = os.path.abspath(self.weights_path)
        self.context_margin = context_margin
        self.appearance_embedder = AnonymousAppearanceEmbedder()
        if metadata_path is not None:
            self.metadata_path = metadata_path
        else:
            meta_candidate = os.path.join(os.path.dirname(self.weights_path), "model_metadata.json")
            self.metadata_path = os.path.abspath(meta_candidate)

        # 1. Load model metadata as SINGLE SOURCE OF TRUTH (strict: file must exist and be complete)
        self.metadata: Dict[str, Any] = self._load_model_metadata(self.metadata_path)

        self.model_version = str(self.metadata["model_version"])
        self.spatial_backbone = str(self.metadata["spatial_backbone"]).lower()
        self.temporal_type = str(self.metadata["temporal_model_type"]).upper()
        self.hidden_dim = int(self.metadata["hidden_dim"])
        self.num_layers = int(self.metadata["num_layers"])
        self.sequence_length = int(self.metadata["sequence_length"])
        self.sampling_fps = float(self.metadata["sampling_fps"])
        self.input_size = tuple(int(v) for v in self.metadata["input_size"])
        self.classes: List[str] = list(self.metadata["classes"])
        self.class_mapping: Dict[str, str] = {
            str(k): str(v) for k, v in self.metadata["class_mapping"].items()
        }

        # Validate backbone type is actually supported by the implementation
        if self.spatial_backbone != "resnet18":
            raise RuntimeError(
                f"Unsupported spatial_backbone '{self.spatial_backbone}' in {self.metadata_path}. "
                f"Only 'resnet18' is implemented."
            )
        if self.temporal_type not in ("GRU", "LSTM", "RNN"):
            raise RuntimeError(
                f"Invalid temporal_model_type '{self.temporal_type}' in {self.metadata_path}. "
                f"Must be one of: GRU, LSTM, RNN."
            )
        # Classes must exactly match the class_mapping
        mapped = sorted(self.class_mapping.values())
        if mapped != sorted(self.classes):
            raise RuntimeError(
                f"class_mapping in {self.metadata_path} does not match classes list."
            )
        if self.input_size != (3, 224, 224):
            raise RuntimeError(
                f"Unsupported input_size {self.input_size} in {self.metadata_path}. "
                f"Only (3, 224, 224) is implemented."
            )

        # Cross-check runtime settings against the single source of truth (fail loudly on mismatch)
        if settings.TEMPORAL_MODEL_TYPE.upper() != self.temporal_type:
            raise RuntimeError(
                f"Configuration mismatch: TEMPORAL_MODEL_TYPE='{settings.TEMPORAL_MODEL_TYPE}' "
                f"but model_metadata.json declares temporal_model_type='{self.temporal_type}'. "
                f"model_metadata.json is the single source of truth; fix the .env or the metadata file."
            )
        if int(settings.TEMPORAL_SEQUENCE_LENGTH) != self.sequence_length:
            raise RuntimeError(
                f"Configuration mismatch: TEMPORAL_SEQUENCE_LENGTH={settings.TEMPORAL_SEQUENCE_LENGTH} "
                f"but model_metadata.json declares sequence_length={self.sequence_length}."
            )
        if int(settings.TEMPORAL_HIDDEN_DIM) != self.hidden_dim:
            raise RuntimeError(
                f"Configuration mismatch: TEMPORAL_HIDDEN_DIM={settings.TEMPORAL_HIDDEN_DIM} "
                f"but model_metadata.json declares hidden_dim={self.hidden_dim}."
            )
        if int(settings.TEMPORAL_NUM_LAYERS) != self.num_layers:
            raise RuntimeError(
                f"Configuration mismatch: TEMPORAL_NUM_LAYERS={settings.TEMPORAL_NUM_LAYERS} "
                f"but model_metadata.json declares num_layers={self.num_layers}."
            )
        if abs(float(settings.SAMPLING_FPS) - self.sampling_fps) > 1e-6:
            raise RuntimeError(
                f"Configuration mismatch: SAMPLING_FPS={settings.SAMPLING_FPS} "
                f"but model_metadata.json declares sampling_fps={self.sampling_fps}."
            )

        # Device Selection
        requested_dev = device_name or settings.ML_DEVICE
        if requested_dev == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_dev if requested_dev in ("cuda", "cpu") else "cpu")

        logger.info(
            f"Initializing MLInferenceService: version={self.model_version}, Backbone={self.spatial_backbone}, "
            f"Temporal={self.temporal_type}, HiddenDim={self.hidden_dim}, NumLayers={self.num_layers}, "
            f"SeqLen={self.sequence_length}, SamplingFPS={self.sampling_fps}, Device={self.device}"
        )

        # 2. Strict Spatial + Temporal Model Initialization & Weights Loading
        self.model = ResNet18TemporalModel(
            temporal_type=self.temporal_type,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=len(self.classes)
        )
        # Strictly load checkpoint - fails explicitly if weights missing or mismatched
        self.model.load_trained_weights(self.weights_path)
        self.model.to(self.device)
        self.model.eval()

        # 3. Pipeline Sub-services
        self.preprocessor = preprocessor
        self.sampler = VideoFrameSampler(
            sampling_fps=self.sampling_fps,
            sequence_length=self.sequence_length,
            stride=max(1, self.sequence_length // 2)
        )
        det_mode = settings.CLASSROOM_DETECTOR_MODE.lower()
        if det_mode in ("tiled", "highrecall"):
            dense_weights = settings.CLASSROOM_DETECTOR_WEIGHTS
            if not os.path.isabs(dense_weights):
                dense_weights = os.path.abspath(dense_weights)
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
                context_margin=self.context_margin
            )
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
        else:
            self.tracker = ByteTracker(
                track_thresh=0.30,
                match_thresh=0.80,
                match_thresh_second=0.40,
                max_time_lost_frames=90
            )
        self.sequence_builder = TrackSequenceBuilder(
            sequence_length=self.sequence_length,
            stride=max(1, self.sequence_length // 2)
        )
        self.annotator = VideoAnnotator(output_dir=settings.OUTPUT_VIDEO_DIR)

        self._last_track_results: List[Dict[str, Any]] = []

    def _load_model_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Loads and strictly validates model_metadata.json. No defaults, no fallbacks."""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"model_metadata.json is REQUIRED as the single source of truth for the temporal model "
                f"architecture, but it was not found at '{metadata_path}'."
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse model metadata from '{metadata_path}': {e}") from e

        missing = [k for k in _REQUIRED_METADATA_KEYS if k not in meta]
        if missing:
            raise RuntimeError(
                f"model_metadata.json at '{metadata_path}' is missing required keys: {missing}. "
                f"It is the single source of truth and must fully describe the trained model."
            )
        logger.info(f"Loaded model configuration from '{metadata_path}'")
        return meta

    def get_last_track_results(self) -> List[Dict[str, Any]]:
        """Returns the student track trajectory results from the most recent analysis run."""
        return self._last_track_results

    def _consolidate_track_fragments(
        self,
        track_results: List[Dict[str, Any]],
        max_gap_frames: int = 300,
        iou_threshold: float = 0.20,
        min_track_observations: int = 3,
    ) -> Dict[int, int]:
        """Consolidates tracker ID fragments into seat-level student tracks.

        Fragments are fused with a union-find over pairwise "same-seat" links:
          - Co-timed overlap: two tracks observed at the same sampled frames are
            the same student only when their boxes actually coincide there
            (mean co-timed IoU >= ``iou_threshold``) - side-by-side students
            have IoU ~ 0 and are never linked.
          - Sequential hand-off: a fragment starting shortly after another ends
            links when its first box strongly overlaps the predecessor's last
            box (>= ``iou_threshold``).
        A merged group is emitted as ONE student only when it is dominated by a
        single track (>= 70% of the group's observations); otherwise the group
        kept its separate member IDs (it would fuse two students who merely
        crossed paths). Tiny flicker fragments (< ``min_track_observations``)
        attach to the spatially nearest group without dominance checks.
        The longest-lived member keeps its ID.
        """

        def box_iou(a: List[float], b: List[float]) -> float:
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / union if union > 0 else 0.0

        tracks = [
            t for t in track_results
            if t.get("bounding_box_history")
        ]
        if len(tracks) <= 1:
            return {}

        def span(t):
            h = t["bounding_box_history"]
            return h[0]["frame"], h[-1]["frame"]

        def box_iou(a, b):
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / union if union > 0 else 0.0

        by_id = {t["track_id"]: t for t in tracks}
        ids = [t["track_id"] for t in tracks]
        obs = {t["track_id"]: len(t["bounding_box_history"]) for t in tracks}

        # ---- Union-find over pairwise same-seat links ----
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
                    # Co-timed: same student only if boxes coincide while both seen.
                    fa = {h["frame"]: h["bbox"] for h in ha}
                    shared = [box_iou(fa[f], h["bbox"]) for h in hb if (f := h["frame"]) in fa]
                    if shared and float(np.mean(shared)) >= iou_threshold:
                        union(a["track_id"], b["track_id"])
                else:
                    # Sequential: fragment continues the predecessor's seat.
                    first_b = hb[0]["bbox"]
                    last_a = ha[-1]["bbox"]
                    if box_iou(first_b, last_a) >= iou_threshold:
                        union(a["track_id"], b["track_id"])

        groups = defaultdict(list)
        for i in ids:
            groups[find(i)].append(i)

        # ---- Dominance check: one track must own >= 70% of the group ----
        assignment: Dict[int, int] = {}
        canonical: List[Dict[str, Any]] = []
        for root, members in groups.items():
            members.sort(key=lambda m: (-obs[m], m))
            total = sum(obs[m] for m in members)
            if obs[members[0]] >= 0.7 * total:
                keep = members[0]
                for m in members:
                    assignment[m] = keep
                canon_track = by_id[keep]
                canon_track = dict(canon_track)
                canon_track["group_members"] = list(members)
                canonical.append(canon_track)
            else:
                # Ambiguous group (two students crossing paths): keep separate.
                for m in members:
                    assignment[m] = m
                    entry = dict(by_id[m])
                    entry["group_members"] = [m]
                    canonical.append(entry)

        # ---- Tiny flicker fragments: attach to nearest group spatially ----
        for t in tracks:
            tid = t["track_id"]
            n_frag = obs[tid]
            if n_frag >= min_track_observations or assignment.get(tid, tid) != tid:
                continue
            h = t["bounding_box_history"]
            f_mid = h[len(h) // 2]
            best = None
            for cand in canonical:
                if cand["track_id"] == tid:
                    continue
                c_members = cand["group_members"]
                c_hist = [hh for m in c_members for hh in histories[m]]
                nearest = min(c_hist, key=lambda hh: abs(hh["frame"] - f_mid["frame"]))
                gap = abs(nearest["frame"] - f_mid["frame"])
                if gap <= max_gap_frames and box_iou(f_mid["bbox"], nearest["bbox"]) >= iou_threshold * 0.5:
                    if best is None or gap < best[0]:
                        best = (gap, cand["track_id"])
            if best is not None:
                assignment[tid] = best[1]

        return assignment

    @staticmethod
    def _filter_fragment_tracks(
        merged_histories: Dict[int, List[Dict[str, Any]]],
        min_distinct_frames: int = 10,
        presence_ratio: float = 0.42,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Drops tracker fragments so only genuine seat-level students remain.

        A real student is present for a substantial fraction of the class, while
        camera-pan/occlusion fragments appear only briefly. A track survives when
        it is observed at least ``min_distinct_frames`` DISTINCT sampled frames
        AND covers at least ``presence_ratio`` of the most-observed track's
        distinct-frame count (co-timed duplicate detections of one person at a
        single frame never count — they collapse to one distinct frame).
        """
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

    def analyze_video_full(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        config: Optional[Dict[str, Any]] = None,
        render_output_video: bool = True
    ) -> Dict[str, Any]:
        """
        Executes full multi-student tracking, temporal behaviour classification, and video annotation pipeline.
        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video is corrupt or unreadable.
            RuntimeError: If model inference fails.
        Returns:
            Dict containing:
              - "predictions": List[Dict] (BehaviourResult formatted)
              - "tracks": List[Dict] (StudentTrackResult formatted)
              - "output_video_path": Optional[str] (Rendered annotated MP4 path)
        """
        logger.info(f"Starting ML individual student-track analysis for video: '{video_path}'")
        if progress_callback:
            progress_callback(5)

        # Reset session tracker state
        self.tracker.reset()
        self._last_track_results = []

        # 1. Sample frames from video (raises ValueError on undecodable/corrupted video)
        sampled_frames = self.sampler.extract_frames(video_path)
        if not sampled_frames:
            raise ValueError(f"No decodable frames extracted from video '{video_path}'")

        if progress_callback:
            progress_callback(20)

        # 2. Detect and Track persons across frames
        track_crops: Dict[int, List[Dict[str, Any]]] = {}
        total_frames = len(sampled_frames)

        for idx, item in enumerate(sampled_frames):
            frame = item["frame_data"]
            f_num = item["frame_number"]
            t_sec = item["timestamp_seconds"]

            # Person detection (Ultralytics YOLO)
            detections = self.detector.detect_persons(frame, frame_idx=f_num, timestamp=t_sec)

            if settings.CLASSROOM_TRACKER_MODE.lower() == "dense":
                for detection in detections:
                    detection["appearance"] = self.appearance_embedder(frame, detection["bbox"])

            # Multi-Object Tracking (ByteTrack)
            active_stracks = self.tracker.update(detections, frame_idx=f_num, timestamp_seconds=t_sec)

            # Extract context & aspect-ratio preserved crops for active student tracks
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
                    "confidence": strack.score
                })

            if progress_callback:
                pct = 20 + int((idx + 1) / total_frames * 25)
                progress_callback(min(45, pct))

        # Build trajectory track results
        all_session_tracks = self.tracker.get_all_session_tracks()
        track_results: List[Dict[str, Any]] = []
        for strack in all_session_tracks:
            track_results.append({
                "track_id": strack.track_id,
                "start_frame": strack.start_frame,
                "end_frame": strack.end_frame,
                "bounding_box_history": strack.history
            })

        # Consolidate tracker ID fragments into seat-level student tracks so a
        # single student is ONE Track ID (labels stay stable Student 01..N).
        consolidation = self._consolidate_track_fragments(track_results)
        merged_histories: Dict[int, List[Dict[str, Any]]] = {}
        for t in track_results:
            canonical_id = consolidation.get(t["track_id"], t["track_id"])
            merged_histories.setdefault(canonical_id, []).extend(t["bounding_box_history"])
        # Re-key crops to canonical IDs so sequences aggregate the whole student
        rekeyed_crops: Dict[int, List[Dict[str, Any]]] = {}
        for tid, obs in track_crops.items():
            rekeyed_crops.setdefault(consolidation.get(tid, tid), []).extend(obs)
        track_crops = rekeyed_crops

        # Filter out fragment tracks: a real student must be observed across
        # multiple DISTINCT sampled frames (co-timed duplicate detections of one
        min_frames_req = min(10, max(1, total_frames // 3))
        merged_histories = self._filter_fragment_tracks(merged_histories, min_distinct_frames=min_frames_req)

        # Renumber surviving consolidated tracks sequentially by first appearance
        # so labels are clean Student 01, 02, 03... instead of raw tracker IDs.
        first_frame = {
            cid: min(h["frame"] for h in hs)
            for cid, hs in merged_histories.items()
        }
        renumber = {
            cid: new_id
            for new_id, cid in enumerate(sorted(first_frame, key=lambda c: first_frame[c]), start=1)
        }
        track_results = [
            {
                "track_id": renumber[cid],
                "start_frame": min(h["frame"] for h in hs),
                "end_frame": max(h["frame"] for h in hs),
                "bounding_box_history": sorted(hs, key=lambda h: h["frame"]),
            }
            for cid, hs in merged_histories.items()
        ]
        track_results.sort(key=lambda t: t["track_id"])
        track_crops = {renumber[cid]: obs for cid, obs in track_crops.items() if cid in renumber}
        logger.info(
            f"Kept {len(track_results)} student tracks "
            f"(>= 10 distinct sampled frames)."
        )

        self._last_track_results = track_results
        if consolidation:
            logger.info(
                f"Consolidated {len(consolidation)} tracker fragments into "
                f"{len(track_results)} seat-level student tracks."
            )

        if progress_callback:
            progress_callback(50)

        # 3. Build independent per-track temporal sequences
        track_sequences = self.sequence_builder.build_track_sequences(track_crops)
        if not track_sequences:
            logger.warning(
                f"No student tracks detected in video '{video_path}'. "
                f"Returning zero predictions (this is a legitimate empty result, not a failure)."
            )
            return {
                "predictions": [],
                "tracks": track_results,
                "output_video_path": None
            }

        # 4. Feature extraction and temporal behavior prediction per track
        predictions: List[Dict[str, Any]] = []
        total_track_count = len(track_sequences)
        processed_tracks = 0

        model_label = f"ResNet18_{self.temporal_type}"

        with torch.no_grad():
            for track_id, sequences in sorted(track_sequences.items(), key=lambda x: x[0]):
                student_label = f"Student {track_id:02d}"
                batch_size = 4
                num_seqs = len(sequences)

                for i in range(0, num_seqs, batch_size):
                    batch_seqs = sequences[i:i + batch_size]
                    batch_tensors = []
                    for seq in batch_seqs:
                        # Preprocess list of numpy crops -> (seq_len, 3, 224, 224)
                        seq_tensor = self.preprocessor.preprocess_batch(seq["crops"])
                        batch_tensors.append(seq_tensor)

                    # Batch input: (B, seq_len, 3, 224, 224)
                    batch_input = torch.stack(batch_tensors, dim=0).to(self.device)

                    # Forward pass through ResNet-18 + Temporal RNN/GRU/LSTM
                    logits = self.model(batch_input)  # (B, 5)
                    probabilities = F.softmax(logits, dim=-1).cpu().numpy()
                    top_indices = torch.argmax(logits, dim=-1).cpu().numpy()

                    for j, seq in enumerate(batch_seqs):
                        pred_idx = int(top_indices[j])
                        pred_label = self.classes[pred_idx]
                        confidence = float(probabilities[j][pred_idx])

                        prob_dist = {
                            self.classes[k]: round(float(probabilities[j][k]), 4)
                            for k in range(len(self.classes))
                        }

                        predictions.append({
                            "track_id": track_id,
                            "student_label": student_label,
                            "frame_number": seq["anchor_frame_number"],
                            "timestamp_seconds": seq["anchor_timestamp"],
                            "behaviour_type": pred_label,
                            "confidence": round(confidence, 4),
                            "metadata_json": {
                                "student_label": student_label,
                                "model": model_label,
                                "model_version": self.model_version,
                                "window_start_sec": seq["window_start_time"],
                                "window_end_sec": seq["window_end_time"],
                                "probability_distribution": prob_dist
                            }
                        })

                processed_tracks += 1
                if progress_callback:
                    pct = 50 + int((processed_tracks / total_track_count) * 35)
                    progress_callback(min(85, pct))

        # 5. Render Annotated Output Video
        output_video_path = None
        if render_output_video:
            output_video_path = self.annotator.render_annotated_video(
                video_path=video_path,
                tracks=track_results,
                predictions=predictions
            )

        logger.info(
            f"Completed individual student-track analysis for '{video_path}'. "
            f"Tracks: {len(track_results)}, Predictions: {len(predictions)}, "
            f"Annotated Video: {output_video_path}"
        )

        if progress_callback:
            progress_callback(100)

        return {
            "predictions": predictions,
            "tracks": track_results,
            "output_video_path": output_video_path
        }

    def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible method returning list of behaviour predictions.
        Also populates self._last_track_results.
        """
        full_res = self.analyze_video_full(
            video_path=video_path,
            progress_callback=progress_callback,
            config=config,
            render_output_video=False
        )
        return full_res["predictions"]


# Global inference service instance
_ml_service_instance: Optional[MLInferenceService] = None

# Aliases
MLService = MLInferenceService


def get_ml_service() -> MLInferenceService:
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLInferenceService()
    return _ml_service_instance
