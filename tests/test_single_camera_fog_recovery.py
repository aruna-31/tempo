import pytest
from app.ml.edge import AnonymousTemporalEvent
from app.ml.fog import FogRoomProcessor

def test_single_camera_fog_recovery_stitching():
    """Verify FogRoomProcessor stitches fragmented tracks at same desk across brief occlusions."""
    fog = FogRoomProcessor(room_id="ROOM-101", buffer_window_seconds=30.0, reference_enrollment=50)

    # Simulate Track 1 ending at frame 20 at seat (x=500, y=300)
    # Simulate Track 2 starting at frame 25 at the same seat (x=504, y=302) due to a 5-frame head-turn/occlusion
    fog.session_trajectories = [
        {
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 20,
            "bounding_box_history": [
                {"frame": f, "bbox": [480, 280, 520, 320]} for f in range(0, 21)
            ]
        },
        {
            "track_id": 2,
            "start_frame": 25,
            "end_frame": 45,
            "bounding_box_history": [
                {"frame": f, "bbox": [482, 282, 524, 324]} for f in range(25, 46)
            ]
        }
    ]

    # Associated emitted events
    ev1 = AnonymousTemporalEvent(
        track_id=1, student_label="Student 01", frame_number=20, timestamp_seconds=1.0,
        window_start_sec=0.0, window_end_sec=1.0, bbox=[480, 280, 520, 320], detection_confidence=0.9,
        behaviour_type="Writing", confidence=0.85, probability_distribution={}
    )
    ev2 = AnonymousTemporalEvent(
        track_id=2, student_label="Student 02", frame_number=45, timestamp_seconds=2.25,
        window_start_sec=1.25, window_end_sec=2.25, bbox=[482, 282, 524, 324], detection_confidence=0.9,
        behaviour_type="Writing", confidence=0.88, probability_distribution={}
    )
    fog.session_events = [ev1, ev2]

    # Run Single-Camera Fog Temporal Recovery
    stats = fog.recover_temporal_trajectories(max_gap_frames=10, seat_distance_threshold=50.0)

    # Verify recovery stats
    assert stats["raw_tracks_count"] == 2
    assert stats["recovered_tracks_count"] == 1  # 2 fragmented tracks fused into 1 persistent student
    assert stats["stitched_fragments_count"] == 1
    assert stats["interpolated_frames_count"] == 4  # frames 21, 22, 23, 24 interpolated
    assert stats["fragmentation_reduction_rate"] == 50.0

    # Verify trajectory history contains interpolated frames
    recovered_track = fog.session_trajectories[0]
    assert recovered_track["track_id"] == 1
    assert recovered_track["start_frame"] == 0
    assert recovered_track["end_frame"] == 45
    assert len(recovered_track["bounding_box_history"]) == 46  # 21 original + 4 interpolated + 21 continuation

    # Check interpolated frames are marked
    inter_frames = [h for h in recovered_track["bounding_box_history"] if h.get("interpolated")]
    assert len(inter_frames) == 4
    for h in inter_frames:
        assert h["recovered_by_fog"] is True

    # Verify anonymous identity continuity: ev2 remapped to Student 01
    assert ev2.track_id == 1
    assert ev2.student_label == "Student 01"

    # Verify privacy constraints: zero face recognition, zero permanent identity
    assert not hasattr(ev2, "face_id")
    assert not hasattr(ev2, "biometric_template")
