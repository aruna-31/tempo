"""
TEMPO — Fog Room Intelligence Subsystem
=======================================
Implements the Fog Layer room-level intelligence abstraction:
  1. Ingests anonymous Edge outputs (AnonymousTemporalEvent & Track Trajectories).
  2. Maintains short-term temporal buffers (e.g. rolling 30s event window).
  3. Computes room-level and student-level coverage, observation quality, and uncertainty.
  4. Detects observable room-level temporal patterns and transitions.
  5. Distinguishes reference student enrollment (e.g. 70 students) from automated detections.
  6. Forwards structured summary payload to FastAPI/PostgreSQL persistence layer.
  7. Strict Privacy: Zero face recognition, zero permanent identity, zero biometric tracking.
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.ml.edge import AnonymousTemporalEvent, EdgeProcessingOutput
from app.ml.observation import (
    compute_session_observation_summary,
    compute_track_observation_metrics,
)

logger = logging.getLogger("tempo.ml.fog")


@dataclass
class RoomTemporalEvent:
    """Represents an observable room-level temporal pattern or change point."""
    event_type: str  # e.g. COLLECTIVE_INSTRUCTION_FOCUS, GROUP_DISCUSSION, COLLECTIVE_DISTRACTION
    timestamp_start: float
    timestamp_end: float
    student_count_involved: int
    confidence: float
    description: str
    severity: str = "OBSERVATION"  # OBSERVATION, REVIEW, INFO

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FogRoomSummary:
    """Aggregated room-level summary metrics produced by Fog intelligence node."""
    room_id: str
    active_students_count: int
    observation_coverage: float
    observation_uncertainty: float
    room_temporal_events: List[RoomTemporalEvent]
    activity_distribution: Dict[str, float]
    raw_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FogRoomProcessor:
    """
    Fog Room Intelligence Node Abstraction.

    Operates at the classroom/exam room boundary:
      - Buffers edge events in short-term sliding windows.
      - Fuses multi-student trajectories across time.
      - Calculates observation quality, coverage, and uncertainty.
      - Discovers room-level collective states and change points without identifying individuals.
      - Prepares clean, validated payloads for persistent database ingestion.
    """

    def __init__(
        self,
        room_id: str = "ROOM-01",
        buffer_window_seconds: float = 30.0,
        reference_enrollment: Optional[int] = None,
        expected_students: Optional[int] = None,
        sequence_length: int = 16,
    ):
        self.room_id = room_id
        self.buffer_window_seconds = float(buffer_window_seconds)
        self.reference_enrollment = expected_students if expected_students is not None else (reference_enrollment or 70)
        self.sequence_length = sequence_length

        # Short-term buffer of recent temporal events (rolling window)
        self.short_term_event_buffer: List[AnonymousTemporalEvent] = []
        # Trajectories maintained across the entire session
        self.session_events: List[AnonymousTemporalEvent] = []
        self.session_trajectories: List[Dict[str, Any]] = []
        self.detected_room_events: List[RoomTemporalEvent] = []
        self.fog_recovery_stats: Dict[str, Any] = {}

        logger.info(
            f"Initialized FogRoomProcessor for Room '{self.room_id}': "
            f"BufferWindow={self.buffer_window_seconds}s, RefEnrollment={self.reference_enrollment}"
        )

    def reset(self) -> None:
        """Resets room buffers between sessions."""
        self.short_term_event_buffer.clear()
        self.session_events.clear()
        self.session_trajectories.clear()
        self.detected_room_events.clear()
        self.fog_recovery_stats.clear()

    def ingest_edge_event(self, event: AnonymousTemporalEvent) -> None:
        """Ingests a single streaming anonymous event from Edge and maintains the rolling buffer."""
        self.session_events.append(event)
        self.short_term_event_buffer.append(event)

        # Evict events older than buffer_window_seconds relative to current latest timestamp
        latest_t = event.timestamp_seconds
        cutoff_t = latest_t - self.buffer_window_seconds
        self.short_term_event_buffer = [
            e for e in self.short_term_event_buffer
            if e.timestamp_seconds >= cutoff_t
        ]

    def ingest_edge_output(self, edge_output: EdgeProcessingOutput) -> None:
        """Ingests a complete batch output from EdgeProcessing."""
        self.session_events.extend(edge_output.events)
        self.session_trajectories = edge_output.track_trajectories

        # Populate short-term buffer with recent tail
        if edge_output.events:
            max_t = max(e.timestamp_seconds for e in edge_output.events)
            cutoff_t = max_t - self.buffer_window_seconds
            self.short_term_event_buffer = [
                e for e in edge_output.events
                if e.timestamp_seconds >= cutoff_t
            ]

    def recover_temporal_trajectories(
        self,
        max_gap_frames: int = 30,
        seat_distance_threshold: float = 65.0,
        scale_tolerance: float = 1.8,
    ) -> Dict[str, Any]:
        """
        Single-Camera Fog Temporal Recovery:
        Stitches fragmented anonymous student tracks caused by brief
        occlusions, desk obstructions, or temporary head turns in a single camera view.

        Seat-Consistent Re-association (zero face recognition, zero permanent identity):
        - Evaluates pairs of non-overlapping tracks (A, B) where B starts after A ends.
        - Matches seat proximity (euclidean distance <= seat_distance_threshold).
        - Matches bounding box scale consistency.
        - Fuses Track B into Track A, interpolating intermediate frames.
        - Updates all emitted AnonymousTemporalEvents to the canonical Track A ID.
        """
        if not self.session_trajectories or len(self.session_trajectories) < 2:
            stats = {
                "raw_tracks_count": len(self.session_trajectories),
                "recovered_tracks_count": len(self.session_trajectories),
                "stitched_fragments_count": 0,
                "interpolated_frames_count": 0,
                "fragmentation_reduction_rate": 0.0,
                "single_camera_mode": "MONOCULAR_EDGE_FOG_RECOVERY",
            }
            self.fog_recovery_stats = stats
            return stats

        raw_count = len(self.session_trajectories)
        trajectories = sorted(
            [dict(t) for t in self.session_trajectories],
            key=lambda t: t.get("start_frame", 0),
        )

        id_remap: Dict[int, int] = {}
        stitched_count = 0
        total_interpolated_frames = 0

        modified = True
        while modified:
            modified = False
            best_pair = None
            min_dist = float("inf")

            for i in range(len(trajectories)):
                t_a = trajectories[i]
                end_a = t_a.get("end_frame", 0)
                hist_a = t_a.get("bounding_box_history", [])
                if not hist_a:
                    continue
                last_box = hist_a[-1]["bbox"]
                center_a = (
                    (last_box[0] + last_box[2]) / 2.0,
                    (last_box[1] + last_box[3]) / 2.0,
                )
                area_a = max(1.0, (last_box[2] - last_box[0]) * (last_box[3] - last_box[1]))

                for j in range(len(trajectories)):
                    if i == j:
                        continue
                    t_b = trajectories[j]
                    start_b = t_b.get("start_frame", 0)

                    gap = start_b - end_a
                    if not (1 <= gap <= max_gap_frames):
                        continue

                    hist_b = t_b.get("bounding_box_history", [])
                    if not hist_b:
                        continue
                    first_box = hist_b[0]["bbox"]
                    center_b = (
                        (first_box[0] + first_box[2]) / 2.0,
                        (first_box[1] + first_box[3]) / 2.0,
                    )
                    area_b = max(1.0, (first_box[2] - first_box[0]) * (first_box[3] - first_box[1]))

                    dist = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
                    if dist > seat_distance_threshold:
                        continue

                    ratio = max(area_a, area_b) / min(area_a, area_b)
                    if ratio > scale_tolerance:
                        continue

                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (i, j, gap, last_box, first_box)

            if best_pair is not None:
                i, j, gap, box_a, box_b = best_pair
                t_a = trajectories[i]
                t_b = trajectories[j]

                interpolated_boxes = []
                for step in range(1, gap):
                    inter_frame = t_a["end_frame"] + step
                    alpha = step / float(gap)
                    inter_bbox = [
                        round(box_a[k] + alpha * (box_b[k] - box_a[k]), 2)
                        for k in range(4)
                    ]
                    interpolated_boxes.append({
                        "frame": inter_frame,
                        "bbox": inter_bbox,
                        "interpolated": True,
                        "recovered_by_fog": True,
                    })

                t_a["bounding_box_history"].extend(interpolated_boxes)
                t_a["bounding_box_history"].extend(t_b["bounding_box_history"])
                t_a["end_frame"] = t_b["end_frame"]
                total_interpolated_frames += len(interpolated_boxes)

                old_id = t_b["track_id"]
                canonical_id = t_a["track_id"]
                while canonical_id in id_remap:
                    canonical_id = id_remap[canonical_id]
                id_remap[old_id] = canonical_id

                trajectories.pop(j)
                stitched_count += 1
                modified = True

        self.session_trajectories = trajectories

        if id_remap:
            for ev in self.session_events:
                if ev.track_id in id_remap:
                    canon_id = id_remap[ev.track_id]
                    ev.track_id = canon_id
                    ev.student_label = f"Student {canon_id:02d}"

            for ev in self.short_term_event_buffer:
                if ev.track_id in id_remap:
                    canon_id = id_remap[ev.track_id]
                    ev.track_id = canon_id
                    ev.student_label = f"Student {canon_id:02d}"

        recovered_count = len(self.session_trajectories)
        reduction = (
            round((raw_count - recovered_count) / max(1, raw_count) * 100.0, 1)
            if raw_count > 0 else 0.0
        )

        stats = {
            "raw_tracks_count": raw_count,
            "recovered_tracks_count": recovered_count,
            "stitched_fragments_count": stitched_count,
            "interpolated_frames_count": total_interpolated_frames,
            "fragmentation_reduction_rate": reduction,
            "single_camera_mode": "MONOCULAR_EDGE_FOG_RECOVERY",
        }
        self.fog_recovery_stats = stats
        logger.info(
            f"[Fog Temporal Recovery] Raw Edge Tracks: {raw_count} -> Recovered Unique Students: {recovered_count} "
            f"(Stitched: {stitched_count}, Interpolated Gaps: {total_interpolated_frames} frames, "
            f"Fragmentation Reduced by {reduction}%)"
        )
        return stats

    def process_edge_output(
        self,
        edge_output: EdgeProcessingOutput,
        expected_students: Optional[int] = None,
    ) -> FogRoomSummary:
        """
        Executes end-to-end Fog processing on an EdgeProcessingOutput:
        Ingests events, performs single-camera temporal trajectory recovery,
        detects room-level patterns, calculates coverage & uncertainty,
        and returns a standardized FogRoomSummary.
        """
        if expected_students is not None:
            self.reference_enrollment = expected_students
        self.reset()
        self.ingest_edge_output(edge_output)
        self.recover_temporal_trajectories()
        raw = self.compute_room_intelligence_summary(total_session_frames=edge_output.total_sampled_frames)
        mean_cov = raw.get("mean_detection_coverage")
        if mean_cov is None or mean_cov <= 0:
            mean_cov = raw.get("reference_coverage_ratio", 0.85)
        mean_unc = raw.get("mean_uncertainty", 0.15)
        return FogRoomSummary(
            room_id=self.room_id,
            active_students_count=raw.get("total_detected_students", 0),
            observation_coverage=round(float(mean_cov or 0.85), 3),
            observation_uncertainty=round(float(mean_unc or 0.15), 3),
            room_temporal_events=self.detected_room_events,
            activity_distribution=raw.get("activity_distribution", {}),
            raw_summary=raw,
        )

    def detect_room_temporal_events(
        self,
        events: Optional[List[AnonymousTemporalEvent]] = None,
        window_duration: float = 10.0,
    ) -> List[RoomTemporalEvent]:
        """
        Analyzes multi-student temporal trajectories across time windows.
        Discovers observable collective states:
          - COLLECTIVE_INSTRUCTION_FOCUS: >= 65% of visible students Looking_Toward_Instruction
          - GROUP_DISCUSSION_PHASE: >= 35% of visible students in Peer_Interaction
          - INDEPENDENT_STUDY_WORK: >= 60% of visible students in Reading/Writing
          - COLLECTIVE_DISTRACTION: >= 35% of visible students Looking_Away
        Requires temporal persistence rather than single-frame spikes.
        """
        target_events = events if events is not None else self.session_events
        if not target_events:
            return []

        # Group events into time windows (e.g. 10s bins)
        timestamps = [e.timestamp_seconds for e in target_events]
        min_t, max_t = min(timestamps), max(timestamps)
        detected: List[RoomTemporalEvent] = []

        curr_start = min_t
        while curr_start <= max_t:
            curr_end = curr_start + window_duration
            window_slice = [
                e for e in target_events
                if curr_start <= e.timestamp_seconds <= curr_end
            ]

            if len(window_slice) >= 3:
                # Group by distinct student track in this window
                by_student = defaultdict(list)
                for e in window_slice:
                    by_student[e.track_id].append(e.behaviour_type)

                total_active = len(by_student)
                # Dominant behaviour per student in window
                student_dominant = {
                    tid: Counter(behs).most_common(1)[0][0]
                    for tid, behs in by_student.items()
                }
                counts = Counter(student_dominant.values())

                # Check collective patterns
                focus_ratio = counts.get("Looking_Toward_Instruction", 0) / total_active
                peer_ratio = counts.get("Peer_Interaction", 0) / total_active
                work_ratio = (counts.get("Reading", 0) + counts.get("Writing", 0)) / total_active
                away_ratio = counts.get("Looking_Away", 0) / total_active

                if focus_ratio >= 0.65:
                    detected.append(RoomTemporalEvent(
                        event_type="COLLECTIVE_INSTRUCTION_FOCUS",
                        timestamp_start=round(curr_start, 2),
                        timestamp_end=round(curr_end, 2),
                        student_count_involved=counts.get("Looking_Toward_Instruction", 0),
                        confidence=round(focus_ratio, 3),
                        description=f"{focus_ratio*100:.1f}% of observed students actively engaged toward instruction.",
                        severity="OBSERVATION",
                    ))
                elif peer_ratio >= 0.35:
                    detected.append(RoomTemporalEvent(
                        event_type="GROUP_DISCUSSION_PHASE",
                        timestamp_start=round(curr_start, 2),
                        timestamp_end=round(curr_end, 2),
                        student_count_involved=counts.get("Peer_Interaction", 0),
                        confidence=round(peer_ratio, 3),
                        description=f"Elevated peer interaction ({peer_ratio*100:.1f}% of students) indicating discussion.",
                        severity="OBSERVATION",
                    ))
                elif work_ratio >= 0.60:
                    detected.append(RoomTemporalEvent(
                        event_type="INDEPENDENT_WORK_PHASE",
                        timestamp_start=round(curr_start, 2),
                        timestamp_end=round(curr_end, 2),
                        student_count_involved=counts.get("Reading", 0) + counts.get("Writing", 0),
                        confidence=round(work_ratio, 3),
                        description=f"Active quiet task work ({work_ratio*100:.1f}% reading/writing).",
                        severity="OBSERVATION",
                    ))
                elif away_ratio >= 0.35:
                    detected.append(RoomTemporalEvent(
                        event_type="COLLECTIVE_DISTRACTION",
                        timestamp_start=round(curr_start, 2),
                        timestamp_end=round(curr_end, 2),
                        student_count_involved=counts.get("Looking_Away", 0),
                        confidence=round(away_ratio, 3),
                        description=f"High off-task orientation ({away_ratio*100:.1f}% looking away).",
                        severity="REVIEW",
                    ))

            curr_start += window_duration / 2.0  # 50% overlap

        # De-duplicate consecutive events of identical type
        merged: List[RoomTemporalEvent] = []
        for ev in detected:
            if merged and merged[-1].event_type == ev.event_type and ev.timestamp_start <= merged[-1].timestamp_end:
                merged[-1].timestamp_end = max(merged[-1].timestamp_end, ev.timestamp_end)
                merged[-1].student_count_involved = max(merged[-1].student_count_involved, ev.student_count_involved)
                merged[-1].confidence = max(merged[-1].confidence, ev.confidence)
            else:
                merged.append(ev)

        self.detected_room_events = merged
        return merged

    def compute_room_intelligence_summary(
        self,
        total_session_frames: int,
    ) -> Dict[str, Any]:
        """
        Synthesizes complete Fog Room Intelligence summary:
          - Reference enrollment vs actual detected count.
          - Observation-aware quality metrics across all tracks.
          - Room activity / behaviour distribution.
          - Detected room temporal events.
        """
        # Convert AnonymousTemporalEvent list to prediction-style dicts
        pred_dicts = [e.to_dict() for e in self.session_events]

        # Calculate session observation summary using observation engine
        obs_summary = compute_session_observation_summary(
            track_results=self.session_trajectories,
            total_session_frames=total_session_frames,
            required_sequence_length=self.sequence_length,
            expected_reference_students=self.reference_enrollment,
            behaviour_predictions=pred_dicts,
        )

        # Detect room temporal events
        room_events = self.detect_room_temporal_events()

        # Compute activity distribution across all emitted events
        if self.session_events:
            counts = Counter(e.behaviour_type for e in self.session_events)
            total = len(self.session_events)
            dist = {k: round(v / total, 4) for k, v in counts.items()}
        else:
            dist = {}

        return {
            "room_id": self.room_id,
            "total_detected_students": obs_summary["total_detected_tracks"],
            "stable_tracks_count": obs_summary["stable_tracks_count"],
            "temporal_ready_tracks_count": obs_summary["temporal_ready_tracks_count"],
            "back_row_tracks_count": obs_summary["back_row_tracks_count"],
            "mean_detection_coverage": obs_summary["mean_detection_coverage"],
            "mean_tracking_coverage": obs_summary["mean_tracking_coverage"],
            "mean_visibility_quality": obs_summary["mean_visibility_quality"],
            "mean_uncertainty": obs_summary["mean_uncertainty"],
            "reference_enrollment": obs_summary.get("reference_student_count"),
            "reference_coverage_ratio": obs_summary.get("reference_coverage_ratio"),
            "reference_note": obs_summary.get("reference_note"),
            "activity_distribution": dist,
            "observation_warnings": obs_summary["observation_warnings"],
            "room_temporal_events": [e.to_dict() for e in room_events],
            "short_term_buffer_event_count": len(self.short_term_event_buffer),
            "fog_recovery_stats": getattr(self, "fog_recovery_stats", {}),
        }

    def prepare_cloud_payload(
        self,
        total_session_frames: int,
        output_video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Packages processed Edge & Fog intelligence into standardized payload
        for FastAPI endpoints and PostgreSQL persistence models:
          - predictions: formatted for BehaviourResult
          - tracks: formatted for StudentTrackResult
          - observation_summary: Fog room summary
          - temporal_events: Room events
        """
        fog_summary = self.compute_room_intelligence_summary(total_session_frames)

        # Format predictions matching BehaviourResult schema
        predictions: List[Dict[str, Any]] = []
        for e in self.session_events:
            predictions.append({
                "track_id": e.track_id,
                "student_label": e.student_label,
                "frame_number": e.frame_number,
                "timestamp_seconds": e.timestamp_seconds,
                "behaviour_type": e.behaviour_type,
                "confidence": e.confidence,
                "metadata_json": {
                    "student_label": e.student_label,
                    "model": e.model,
                    "model_version": e.model_version,
                    "window_start_sec": e.window_start_sec,
                    "window_end_sec": e.window_end_sec,
                    "probability_distribution": e.probability_distribution,
                    "attention_weights": e.attention_weights,
                    "is_back_row": e.is_back_row,
                    "detection_confidence": e.detection_confidence,
                    "room_id": self.room_id,
                },
            })

        # Enrich tracks with observation quality metrics
        enriched_tracks: List[Dict[str, Any]] = []
        for t in self.session_trajectories:
            t_copy = dict(t)
            t_copy["observation_quality"] = compute_track_observation_metrics(
                track=t,
                total_session_frames=total_session_frames,
                required_sequence_length=self.sequence_length,
                behaviour_predictions=predictions,
            )
            enriched_tracks.append(t_copy)

        return {
            "predictions": predictions,
            "tracks": enriched_tracks,
            "output_video_path": output_video_path,
            "observation_summary": fog_summary,
            "temporal_events": fog_summary["room_temporal_events"],
        }
