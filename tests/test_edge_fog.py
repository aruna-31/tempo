import os
import pytest
from app.core.config import settings
from app.ml.edge import AnonymousTemporalEvent, EdgeProcessing, EdgeProcessingOutput
from app.ml.fog import FogRoomProcessor, RoomTemporalEvent
from app.ml.model import CLASS_NAMES
from app.ml.service import MLInferenceService


@pytest.fixture
def sample_video_path() -> str:
    path = os.path.abspath("storage/test_videos/classroom_real_persons.mp4")
    if not os.path.exists(path):
        pytest.skip(f"Sample test video not found at {path}")
    return path


def test_edge_processing_initialization():
    """Verify EdgeProcessing initializes components strictly from model metadata."""
    edge = EdgeProcessing()
    assert edge.spatial_backbone == "resnet50"
    assert edge.temporal_type == "BIGRU"
    assert edge.hidden_dim == 256
    assert edge.num_layers == 2
    assert edge.sequence_length == 16
    assert edge.detector is not None
    assert edge.tracker is not None


def test_edge_processing_video_and_event_emission(sample_video_path):
    """Verify EdgeProcessing runs acquisition, detection, tracking, inference, and emits anonymous events."""
    edge = EdgeProcessing()
    output = edge.process_video(sample_video_path)

    assert isinstance(output, EdgeProcessingOutput)
    assert output.session_id is not None
    assert output.total_sampled_frames > 0
    assert output.video_duration_seconds > 0

    # Verify emitted events
    for event in output.events:
        assert isinstance(event, AnonymousTemporalEvent)
        assert isinstance(event.track_id, int)
        assert event.student_label.startswith("Student ")
        assert event.behaviour_type in CLASS_NAMES
        assert 0.0 <= event.confidence <= 1.0
        assert len(event.probability_distribution) == 5
        assert len(event.bbox) == 4
        # Verify strict privacy: no facial feature vectors or permanent identity
        assert not hasattr(event, "face_embedding")
        assert not hasattr(event, "biometric_id")


def test_fog_room_processor_buffering_and_eviction():
    """Verify FogRoomProcessor maintains short-term rolling buffer and evicts old events."""
    fog = FogRoomProcessor(room_id="ROOM-101", buffer_window_seconds=10.0, reference_enrollment=70)

    # Ingest event at t = 5.0s
    e1 = AnonymousTemporalEvent(
        track_id=1, student_label="Student 01", frame_number=10, timestamp_seconds=5.0,
        window_start_sec=1.0, window_end_sec=5.0, bbox=[10, 10, 50, 50], detection_confidence=0.9,
        behaviour_type="Looking_Toward_Instruction", confidence=0.85, probability_distribution={},
    )
    fog.ingest_edge_event(e1)
    assert len(fog.short_term_event_buffer) == 1

    # Ingest event at t = 20.0s (cutoff = 20 - 10 = 10s, so e1 at 5.0s is evicted from short-term buffer)
    e2 = AnonymousTemporalEvent(
        track_id=1, student_label="Student 01", frame_number=40, timestamp_seconds=20.0,
        window_start_sec=16.0, window_end_sec=20.0, bbox=[10, 10, 50, 50], detection_confidence=0.9,
        behaviour_type="Reading", confidence=0.88, probability_distribution={},
    )
    fog.ingest_edge_event(e2)
    assert len(fog.short_term_event_buffer) == 1
    assert fog.short_term_event_buffer[0].timestamp_seconds == 20.0
    # Session events retain history
    assert len(fog.session_events) == 2


def test_fog_room_processor_room_temporal_event_detection():
    """Verify FogRoomProcessor detects collective instructional and discussion phases."""
    fog = FogRoomProcessor(room_id="ROOM-101", buffer_window_seconds=30.0)

    # Generate events where 4 out of 4 students are Looking_Toward_Instruction (100% focus)
    synthetic_events = [
        AnonymousTemporalEvent(
            track_id=tid, student_label=f"Student {tid:02d}", frame_number=10, timestamp_seconds=5.0,
            window_start_sec=1.0, window_end_sec=5.0, bbox=[10, 10, 50, 50], detection_confidence=0.9,
            behaviour_type="Looking_Toward_Instruction", confidence=0.85, probability_distribution={},
        )
        for tid in [1, 2, 3, 4]
    ]

    events = fog.detect_room_temporal_events(events=synthetic_events, window_duration=10.0)
    assert len(events) >= 1
    assert events[0].event_type == "COLLECTIVE_INSTRUCTION_FOCUS"
    assert events[0].severity == "OBSERVATION"


def test_fog_room_processor_prepare_cloud_payload():
    """Verify cloud payload formatting for FastAPI / PostgreSQL layer."""
    fog = FogRoomProcessor(room_id="ROOM-202", reference_enrollment=70)
    e = AnonymousTemporalEvent(
        track_id=1, student_label="Student 01", frame_number=10, timestamp_seconds=5.0,
        window_start_sec=1.0, window_end_sec=5.0, bbox=[10, 10, 50, 50], detection_confidence=0.9,
        behaviour_type="Writing", confidence=0.82, probability_distribution={"Writing": 0.82},
    )
    fog.session_trajectories = [
        {"track_id": 1, "start_frame": 0, "end_frame": 20, "bounding_box_history": [{"frame": 10, "bbox": [10, 10, 50, 50]}]}
    ]
    fog.ingest_edge_event(e)

    payload = fog.prepare_cloud_payload(total_session_frames=20, output_video_path="storage/out.mp4")
    assert "predictions" in payload
    assert "tracks" in payload
    assert "observation_summary" in payload
    assert "temporal_events" in payload

    # Verify reference enrollment separation
    obs = payload["observation_summary"]
    assert obs["reference_enrollment"] == 70
    assert "NOT automated ground truth" in obs["reference_note"]

    # Verify track has observation quality metrics
    track = payload["tracks"][0]
    assert "observation_quality" in track
    assert "detection_coverage" in track["observation_quality"]


def test_ml_service_local_mode_end_to_end(sample_video_path):
    """Verify MLInferenceService seamlessly executes in LOCAL mode via EdgeProcessing -> FogRoomProcessor."""
    service = MLInferenceService()
    assert service.processing_mode == "LOCAL"
    assert service.edge_processor is not None
    assert service.fog_processor is not None

    result = service.analyze_video_full(sample_video_path, render_output_video=False)
    assert "predictions" in result
    assert "tracks" in result
    assert "observation_summary" in result
    assert isinstance(result["predictions"], list)
    assert isinstance(result["tracks"], list)
    assert len(service.get_last_track_results()) == len(result["tracks"])
