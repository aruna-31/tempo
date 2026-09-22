import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.analysis import (
    BehaviourResult,
    BehaviourTransition,
    ChangePoint,
    ClassroomEntropy,
    ClassroomTemporalState,
    CoverageMetric,
    StudentTemporalProfile,
    StudentTrackResult,
)

BEHAVIOURS = [
    "Looking_Toward_Instruction",
    "Reading",
    "Writing",
    "Peer_Interaction",
    "Looking_Away",
]


def segment_labels(predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(predictions, key=lambda item: (item["timestamp_seconds"], item["frame_number"]))
    segments: List[Dict[str, Any]] = []
    for prediction in ordered:
        label = prediction["behaviour_type"]
        timestamp = float(prediction["timestamp_seconds"])
        if segments and segments[-1]["behaviour"] == label:
            segment = segments[-1]
            segment["end_time"] = timestamp
            segment["end_frame"] = prediction["frame_number"]
            segment["confidence_sum"] += float(prediction["confidence"])
            segment["count"] += 1
        else:
            segments.append({
                "behaviour": label,
                "start_time": timestamp,
                "end_time": timestamp,
                "start_frame": prediction["frame_number"],
                "end_frame": prediction["frame_number"],
                "confidence_sum": float(prediction["confidence"]),
                "count": 1,
            })
    for segment in segments:
        segment["confidence"] = round(segment.pop("confidence_sum") / segment.pop("count"), 4)
        segment["duration_seconds"] = max(0.0, segment["end_time"] - segment["start_time"])
    return segments


def transition_matrix(labels: Sequence[str]) -> Tuple[List[List[int]], List[List[float]], List[Dict[str, str]]]:
    counts = [[0 for _ in BEHAVIOURS] for _ in BEHAVIOURS]
    segments = segment_labels([
        {"behaviour_type": label, "timestamp_seconds": index, "frame_number": index, "confidence": 1.0}
        for index, label in enumerate(labels)
    ])
    collapsed = [segment["behaviour"] for segment in segments]
    index = {label: position for position, label in enumerate(BEHAVIOURS)}
    for previous, current in zip(collapsed, collapsed[1:]):
        counts[index[previous]][index[current]] += 1
    probabilities = []
    for row in counts:
        total = sum(row)
        probabilities.append([round(value / total, 6) if total else 0.0 for value in row])
    transitions = [
        {"from": BEHAVIOURS[row], "to": BEHAVIOURS[column], "count": counts[row][column], "probability": probabilities[row][column]}
        for row in range(len(BEHAVIOURS))
        for column in range(len(BEHAVIOURS))
        if counts[row][column]
    ]
    return counts, probabilities, transitions


def shannon_entropy(distribution: Dict[str, float]) -> float:
    return round(-sum(value * math.log(value) for value in distribution.values() if value > 0), 6)


def jensen_shannon_divergence(first: Dict[str, float], second: Dict[str, float]) -> float:
    midpoint = {label: (first.get(label, 0.0) + second.get(label, 0.0)) / 2 for label in BEHAVIOURS}
    return round((shannon_entropy(midpoint) - shannon_entropy(first) / 2 - shannon_entropy(second) / 2), 6)


def classroom_state(distribution: Dict[str, float]) -> str:
    dominant = max(distribution, key=distribution.get) if distribution else None
    if not dominant or distribution[dominant] < 0.4:
        return "Mixed-Activity"
    return {
        "Looking_Toward_Instruction": "Instruction-Dominant",
        "Reading": "Reading-Dominant",
        "Writing": "Individual-Work-Dominant",
        "Peer_Interaction": "Collaborative-Interaction-Dominant",
        "Looking_Away": "Mixed-Activity",
    }[dominant]


class TemporalAnalyticsService:
    @staticmethod
    def _job_rows(db: Session, job_id: UUID):
        tracks = db.query(StudentTrackResult).filter(StudentTrackResult.job_id == job_id).all()
        behaviours = db.query(BehaviourResult).filter(BehaviourResult.job_id == job_id).order_by(BehaviourResult.timestamp_seconds.asc()).all()
        return tracks, behaviours

    @staticmethod
    def build_and_persist(db: Session, job_id: UUID, window_seconds: float = 60.0, manual_reference_student_count: int | None = None) -> Dict[str, Any]:
        tracks, behaviours = TemporalAnalyticsService._job_rows(db, job_id)
        by_track: Dict[int, List[BehaviourResult]] = defaultdict(list)
        for result in behaviours:
            if result.behaviour_type in BEHAVIOURS:
                by_track[result.track_id].append(result)

        for model in (StudentTemporalProfile, BehaviourTransition, ClassroomTemporalState, ClassroomEntropy, ChangePoint, CoverageMetric):
            db.query(model).filter(model.job_id == job_id).delete(synchronize_session=False)

        profile_payloads = []
        all_transitions = Counter()
        all_observations: List[Tuple[float, str, int]] = []
        temporal_ready = 0
        for track in tracks:
            rows = by_track.get(track.track_id, [])
            segments = segment_labels([
                {"behaviour_type": row.behaviour_type, "timestamp_seconds": row.timestamp_seconds, "frame_number": row.frame_number, "confidence": row.confidence}
                for row in rows
            ])
            labels = [row.behaviour_type for row in rows]
            counts, probabilities, transitions = transition_matrix(labels)
            for transition in transitions:
                all_transitions[(transition["from"], transition["to"])] += transition["count"]
            for row in rows:
                all_observations.append((row.timestamp_seconds, row.behaviour_type, row.track_id))
            total_duration = max((item["end_time"] for item in segments), default=0.0) - min((item["start_time"] for item in segments), default=0.0)
            duration_distribution = {label: round(sum(item["duration_seconds"] for item in segments if item["behaviour"] == label), 4) for label in BEHAVIOURS}
            total = len(rows)
            behaviour_distribution = {label: round(sum(1 for item in labels if item == label) / total, 6) if total else 0.0 for label in BEHAVIOURS}
            temporal_coverage = round(len(rows) / max(1, len(track.bounding_box_history)), 6)
            if len(rows) >= 2:
                temporal_ready += 1
            profile_payloads.append(StudentTemporalProfile(
                job_id=job_id, track_id=track.track_id, timeline_json=segments,
                behaviour_distribution_json=behaviour_distribution, transition_matrix_json=probabilities,
                total_observed_duration=round(total_duration, 4), behaviour_duration_json=duration_distribution,
                segment_count=len(segments), temporal_coverage=temporal_coverage,
            ))
        db.add_all(profile_payloads)
        db.add_all([
            BehaviourTransition(job_id=job_id, track_id=None, from_behaviour=source, to_behaviour=target, count=count, probability=round(count / sum(value for (s, t), value in all_transitions.items() if s == source), 6))
            for (source, target), count in all_transitions.items()
        ])

        max_time = max((item[0] for item in all_observations), default=0.0)
        states = []
        for start in range(0, int(max_time) + 1, int(window_seconds)):
            window = [item for item in all_observations if start <= item[0] < start + window_seconds]
            counts = Counter(item[1] for item in window)
            total = sum(counts.values())
            if not total:
                continue
            distribution = {label: round(counts[label] / total, 6) for label in BEHAVIOURS}
            state = classroom_state(distribution)
            state_model = ClassroomTemporalState(job_id=job_id, start_time=float(start), end_time=float(start + window_seconds), state=state, behaviour_distribution_json=distribution, students_contributing=len({item[2] for item in window}))
            entropy_model = ClassroomEntropy(job_id=job_id, timestamp=float(start), entropy=shannon_entropy(distribution), behaviour_distribution_json=distribution)
            states.append((state_model, entropy_model, distribution, state))
        db.add_all([item[0] for item in states]); db.add_all([item[1] for item in states])
        for previous, current in zip(states, states[1:]):
            score = jensen_shannon_divergence(previous[2], current[2])
            if score > 0.1:
                db.add(ChangePoint(job_id=job_id, timestamp=current[0].start_time, previous_distribution_json=previous[2], new_distribution_json=current[2], change_score=score, previous_state=previous[3], new_state=current[3]))

        detected = len(tracks)
        stable = sum(1 for track in tracks if len({item.get("frame") for item in (track.bounding_box_history or [])}) >= 3)
        detection_coverage = round(detected / manual_reference_student_count, 6) if manual_reference_student_count else None
        db.add(CoverageMetric(job_id=job_id, manual_reference_student_count=manual_reference_student_count, detected_student_count=detected, tracked_student_count=stable, temporal_ready_track_count=temporal_ready, detection_coverage=detection_coverage, tracking_coverage=round(stable / detected, 6) if detected else 0.0, temporal_coverage=round(temporal_ready / stable, 6) if stable else 0.0))
        db.commit()
        return TemporalAnalyticsService.read_all(db, job_id)

    @staticmethod
    def read_all(db: Session, job_id: UUID) -> Dict[str, Any]:
        def profile(item):
            return {"track_id": item.track_id, "timeline": item.timeline_json, "behaviour_distribution": item.behaviour_distribution_json, "transition_matrix": item.transition_matrix_json, "behaviour_duration": item.behaviour_duration_json, "total_observed_duration": item.total_observed_duration, "segment_count": item.segment_count, "temporal_coverage": item.temporal_coverage}
        def transition(item):
            return {"track_id": item.track_id, "from_behaviour": item.from_behaviour, "to_behaviour": item.to_behaviour, "count": item.count, "probability": item.probability}
        def state(item):
            return {"start_time": item.start_time, "end_time": item.end_time, "state": item.state, "behaviour_distribution": item.behaviour_distribution_json, "students_contributing": item.students_contributing}
        def entropy(item):
            return {"timestamp": item.timestamp, "entropy": item.entropy, "behaviour_distribution": item.behaviour_distribution_json}
        def change(item):
            return {"timestamp": item.timestamp, "previous_distribution": item.previous_distribution_json, "new_distribution": item.new_distribution_json, "change_score": item.change_score, "previous_state": item.previous_state, "new_state": item.new_state}
        def coverage(item):
            if not item:
                return None
            return {"manual_reference_student_count": item.manual_reference_student_count, "detected_student_count": item.detected_student_count, "tracked_student_count": item.tracked_student_count, "temporal_ready_track_count": item.temporal_ready_track_count, "detection_coverage": item.detection_coverage, "tracking_coverage": item.tracking_coverage, "temporal_coverage": item.temporal_coverage}
        return {
            "students": [profile(item) for item in db.query(StudentTemporalProfile).filter_by(job_id=job_id).order_by(StudentTemporalProfile.track_id).all()],
            "transitions": [transition(item) for item in db.query(BehaviourTransition).filter_by(job_id=job_id).all()],
            "classroom_states": [state(item) for item in db.query(ClassroomTemporalState).filter_by(job_id=job_id).order_by(ClassroomTemporalState.start_time).all()],
            "entropy": [entropy(item) for item in db.query(ClassroomEntropy).filter_by(job_id=job_id).order_by(ClassroomEntropy.timestamp).all()],
            "change_points": [change(item) for item in db.query(ChangePoint).filter_by(job_id=job_id).order_by(ChangePoint.timestamp).all()],
            "coverage": coverage(db.query(CoverageMetric).filter_by(job_id=job_id).first()),
        }