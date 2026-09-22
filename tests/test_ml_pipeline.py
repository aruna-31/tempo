import datetime
import os
import cv2
import numpy as np
import pytest
import torch
from sqlalchemy.orm import Session
from app.ml.detector import YOLOPersonDetector
from app.ml.model import CLASS_NAMES, ResNet18TemporalModel
from app.ml.preprocessor import FramePreprocessor
from app.ml.sampler import TrackSequenceBuilder, VideoFrameSampler
from app.ml.service import MLInferenceService
from app.ml.tracker import ByteTracker, STrack
from app.models.academic import Section, Subject
from app.models.analysis import AnalysisJob, BehaviourResult, StudentTrackResult
from app.models.faculty import Faculty
from app.models.session import ClassSession
from app.models.video import Video
from app.services.ml_connector import run_video_analysis_worker


@pytest.fixture
def sample_real_video_path(tmp_path) -> str:
    """
    Creates a real valid decodable MP4 video containing real persons for end-to-end testing.
    """
    real_mp4 = os.path.abspath("storage/test_videos/classroom_real_persons.mp4")
    if os.path.exists(real_mp4):
        return real_mp4

    bus_path = "bus.jpg"
    if os.path.exists(bus_path):
        base_img = cv2.imread(bus_path)
    else:
        base_img = np.full((480, 640, 3), 200, dtype=np.uint8)

    h, w = base_img.shape[:2]
    video_file = os.path.join(tmp_path, "test_real_classroom.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_file, fourcc, 30.0, (w, h))

    for f in range(60):  # 2 seconds at 30 fps
        dx = int(np.sin(f * 0.1) * 2)
        dy = int(np.cos(f * 0.1) * 2)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(base_img, M, (w, h))
        writer.write(shifted)

    writer.release()
    return video_file


def test_resnet18_temporal_model_architectures():
    batch_size = 2
    seq_len = 8
    # Input tensor: (B, seq_len, 3, 224, 224)
    dummy_input = torch.randn(batch_size, seq_len, 3, 224, 224)

    for temporal_type in ["GRU", "LSTM", "RNN"]:
        model = ResNet18TemporalModel(
            temporal_type=temporal_type,
            hidden_dim=128,
            num_layers=1,
            num_classes=5
        )
        model.eval()
        with torch.no_grad():
            output_logits = model(dummy_input)

        assert output_logits.shape == (batch_size, 5), (
            f"Expected output shape ({batch_size}, 5) for {temporal_type}, got {output_logits.shape}"
        )


def test_strict_model_weights_loading():
    # Test valid metadata loading
    model = ResNet18TemporalModel(
        temporal_type="RNN",
        hidden_dim=128,
        num_layers=2,
        num_classes=5
    )
    loaded = model.load_trained_weights("models/classroom_temporal_model.pth")
    assert loaded is True

    # Test error when file does not exist
    with pytest.raises(FileNotFoundError):
        model.load_trained_weights("models/non_existent_weights.pth")

    # Test architecture mismatch raises RuntimeError
    mismatched_model = ResNet18TemporalModel(
        temporal_type="LSTM",
        hidden_dim=256,
        num_layers=2,
        num_classes=5
    )
    with pytest.raises(RuntimeError):
        mismatched_model.load_trained_weights("models/classroom_temporal_model.pth")


def test_frame_preprocessor():
    preprocessor = FramePreprocessor()
    numpy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor = preprocessor.preprocess_numpy_frame(numpy_frame)

    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)

    batch_tensor = preprocessor.preprocess_batch([numpy_frame, numpy_frame])
    assert batch_tensor.shape == (2, 3, 224, 224)


def test_yolo_person_detector_and_context_crop():
    detector = YOLOPersonDetector(confidence_threshold=0.25, context_margin=0.12)
    
    # Test on real image with people
    real_mp4 = os.path.abspath("storage/test_videos/classroom_real_persons.mp4")
    if os.path.exists("bus.jpg"):
        test_frame = cv2.imread("bus.jpg")
    elif os.path.exists(real_mp4):
        cap = cv2.VideoCapture(real_mp4)
        ret, test_frame = cap.read()
        cap.release()
    else:
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    detections = detector.detect_persons(test_frame, frame_idx=0, timestamp=0.0)
    assert len(detections) >= 1
    for det in detections:
        assert "bbox" in det
        assert len(det["bbox"]) == 4
        assert det["confidence"] > 0.0
        assert det["class_name"] == "person"

    # Test aspect-ratio preserved context crop extraction
    first_bbox = detections[0]["bbox"]
    crop = detector.extract_person_crop(test_frame, first_bbox, target_size=(224, 224))
    assert crop.shape == (224, 224, 3)
    assert isinstance(crop, np.ndarray)


def test_yolo_detector_invalid_input_error():
    detector = YOLOPersonDetector(confidence_threshold=0.30)
    with pytest.raises(ValueError):
        detector.detect_persons(None)

    with pytest.raises(ValueError):
        detector.detect_persons(np.empty((0, 0, 3), dtype=np.uint8))


def test_bytetrack_multi_object_tracking():
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.8)
    tracker.reset()

    # Frame 0: Detect 2 students
    dets_f0 = [
        {"bbox": [50.0, 100.0, 150.0, 300.0], "confidence": 0.90},
        {"bbox": [250.0, 100.0, 350.0, 300.0], "confidence": 0.88}
    ]
    active_tracks_f0 = tracker.update(dets_f0, frame_idx=0, timestamp_seconds=0.0)
    assert len(active_tracks_f0) == 2
    track_ids_f0 = [t.track_id for t in active_tracks_f0]
    assert track_ids_f0 == [1, 2]

    # Frame 1: Same students with small movement
    dets_f1 = [
        {"bbox": [52.0, 101.0, 152.0, 301.0], "confidence": 0.92},
        {"bbox": [251.0, 99.0, 351.0, 299.0], "confidence": 0.85}
    ]
    active_tracks_f1 = tracker.update(dets_f1, frame_idx=1, timestamp_seconds=0.5)
    assert len(active_tracks_f1) == 2
    track_ids_f1 = [t.track_id for t in active_tracks_f1]
    assert track_ids_f1 == [1, 2]

    # Frame 2: Student 2 temporarily occluded
    dets_f2 = [
        {"bbox": [53.0, 102.0, 153.0, 302.0], "confidence": 0.91}
    ]
    active_tracks_f2 = tracker.update(dets_f2, frame_idx=2, timestamp_seconds=1.0)
    assert len(active_tracks_f2) == 1
    assert active_tracks_f2[0].track_id == 1

    # Frame 3: Student 2 reappears (re-activated)
    dets_f3 = [
        {"bbox": [54.0, 102.0, 154.0, 302.0], "confidence": 0.93},
        {"bbox": [252.0, 100.0, 352.0, 300.0], "confidence": 0.87}
    ]
    active_tracks_f3 = tracker.update(dets_f3, frame_idx=3, timestamp_seconds=1.5)
    assert len(active_tracks_f3) == 2
    reappeared_ids = sorted([t.track_id for t in active_tracks_f3])
    assert reappeared_ids == [1, 2]


def test_video_frame_sampler_strict_errors(tmp_path):
    sampler = VideoFrameSampler(sampling_fps=2.0)

    # Missing file
    with pytest.raises(FileNotFoundError):
        sampler.extract_frames("storage/non_existent_video.mp4")

    # Corrupt/undecodable stub
    corrupt_file = os.path.join(tmp_path, "corrupt.mp4")
    with open(corrupt_file, "wb") as f:
        f.write(b"CORRUPT_NOT_A_REAL_VIDEO_STREAM")

    with pytest.raises(ValueError):
        sampler.extract_frames(corrupt_file)


def test_track_sequence_builder():
    builder = TrackSequenceBuilder(sequence_length=4, stride=2)
    fake_crop = np.zeros((224, 224, 3), dtype=np.uint8)

    track_crops = {
        1: [
            {"frame_number": i * 15, "timestamp_seconds": i * 0.5, "crop": fake_crop, "bbox": [10, 10, 50, 50], "confidence": 0.9}
            for i in range(8)
        ],
        2: [
            {"frame_number": i * 15, "timestamp_seconds": i * 0.5, "crop": fake_crop, "bbox": [100, 10, 150, 50], "confidence": 0.85}
            for i in range(5)
        ]
    }

    sequences_map = builder.build_track_sequences(track_crops)
    assert 1 in sequences_map
    assert 2 in sequences_map
    assert len(sequences_map[1]) > 0
    assert len(sequences_map[1][0]["crops"]) == 4
    assert sequences_map[1][0]["track_id"] == 1


def test_ml_inference_service_end_to_end_real_video(sample_real_video_path):
    service = MLInferenceService(device_name="cpu")

    progress_records = []

    def on_progress(p):
        progress_records.append(p)

    output = service.analyze_video_full(
        video_path=sample_real_video_path,
        progress_callback=on_progress,
        render_output_video=True
    )

    predictions = output["predictions"]
    tracks = output["tracks"]
    output_video_path = output["output_video_path"]

    assert len(tracks) >= 1
    assert len(predictions) >= 1
    assert len(progress_records) > 0

    # Verify per-student track predictions
    for p in predictions:
        assert p["behaviour_type"] in CLASS_NAMES
        assert 0.0 <= p["confidence"] <= 1.0
        assert "probability_distribution" in p["metadata_json"]
        assert len(p["metadata_json"]["probability_distribution"]) == 5
        assert p["track_id"] in [t["track_id"] for t in tracks]

    # Verify rendered annotated video exists on disk and is non-empty
    assert output_video_path is not None
    assert os.path.exists(output_video_path)
    assert os.path.getsize(output_video_path) > 0

    # Verify output video is decodable by OpenCV
    cap = cv2.VideoCapture(output_video_path)
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret is True
    assert frame is not None
    cap.release()


def test_ml_worker_database_execution_with_student_tracks(
    db_session: Session, faculty_a: Faculty, sample_real_video_path
):
    subject = Subject(faculty_id=faculty_a.id, code="23CS8888", name="ML Test Subject")
    db_session.add(subject)
    db_session.commit()

    section = Section(subject_id=subject.id, name="Sec-ML", academic_year="2025-2026", semester="Semester 5")
    db_session.add(section)
    db_session.commit()

    session = ClassSession(
        faculty_id=faculty_a.id,
        section_id=section.id,
        title="ML Lab Session",
        session_date=datetime.date(2026, 9, 19),
        start_time=datetime.time(10, 0),
        end_time=datetime.time(11, 0)
    )
    db_session.add(session)
    db_session.commit()

    video = Video(
        session_id=session.id,
        faculty_id=faculty_a.id,
        file_path=sample_real_video_path,
        original_filename="real_test.mp4",
        file_size_bytes=os.path.getsize(sample_real_video_path),
        status="UPLOADED"
    )
    db_session.add(video)
    db_session.commit()

    job = AnalysisJob(
        video_id=video.id,
        status="PENDING",
        progress_pct=0,
        config_json={"model": "ResNet18_RNN"}
    )
    db_session.add(job)
    db_session.commit()

    # Run analysis worker synchronously
    run_video_analysis_worker(job.id)

    db_session.refresh(job)
    db_session.refresh(video)

    assert job.status == "COMPLETED"
    assert job.progress_pct == 100
    assert video.status == "ANALYZED"
    assert video.output_video_path is not None
    assert os.path.exists(video.output_video_path)

    saved_tracks = db_session.query(StudentTrackResult).filter(StudentTrackResult.job_id == job.id).all()
    assert len(saved_tracks) >= 1
    assert saved_tracks[0].track_id == 1
    assert len(saved_tracks[0].bounding_box_history) > 0

    saved_behaviours = db_session.query(BehaviourResult).filter(BehaviourResult.job_id == job.id).all()
    assert len(saved_behaviours) > 0
    assert saved_behaviours[0].behaviour_type in CLASS_NAMES
    assert saved_behaviours[0].track_id in [t.track_id for t in saved_tracks]


# ---------------------------------------------------------------------------
# Track fragment consolidation (one stable Student ID per seated student)
# ---------------------------------------------------------------------------
def _hist(track_id, start, n, bbox):
    return {
        "track_id": track_id,
        "start_frame": start,
        "end_frame": start + (n - 1) * 15,
        "bounding_box_history": [
            {"frame": start + i * 15, "timestamp": (start + i * 15) / 30.0,
             "bbox": list(bbox), "confidence": 0.8}
            for i in range(n)
        ],
    }


def test_fragments_merge_into_seat_track():
    svc = MLInferenceService.__new__(MLInferenceService)  # no model load needed
    long_track = _hist(1, 0, 60, [100, 200, 400, 700])
    frag = _hist(2, 300, 5, [104, 202, 398, 696])
    mapping = svc._consolidate_track_fragments([long_track, frag])
    assert mapping[2] == 1 and mapping[1] == 1


def test_distinct_students_never_merge():
    svc = MLInferenceService.__new__(MLInferenceService)
    left = _hist(1, 0, 60, [100, 200, 300, 600])
    right = _hist(2, 0, 60, [1300, 200, 1500, 600])
    mapping = svc._consolidate_track_fragments([left, right])
    assert mapping[1] == 1 and mapping[2] == 2


def test_single_sample_flicker_absorbed():
    svc = MLInferenceService.__new__(MLInferenceService)
    real = _hist(3, 0, 80, [500, 200, 800, 700])
    flicker = _hist(9, 450, 1, [505, 205, 795, 695])
    mapping = svc._consolidate_track_fragments([real, flicker])
    assert mapping[9] == 3


# ---------------------------------------------------------------------------
# Annotated video must be browser-playable (H.264 in MP4)
# ---------------------------------------------------------------------------
def test_rendered_output_is_h264_mp4(tmp_path):
    src = str(tmp_path / "in.mp4")
    w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(30):
        w.write(np.full((240, 320, 3), 120, np.uint8))
    w.release()

    from app.ml.annotator import VideoAnnotator
    ann = VideoAnnotator(output_dir=str(tmp_path))
    out = ann.render_annotated_video(video_path=src, tracks=[], predictions=[])
    assert os.path.exists(out) and os.path.getsize(out) > 0
    cap = cv2.VideoCapture(out)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
    ok, _ = cap.read()
    cap.release()
    assert ok, "output video must be decodable"
    assert fourcc in ("avc1", "h264", "H264"), f"expected H.264, got {fourcc!r}"


# ---------------------------------------------------------------------------
# Fragment filter: real students must span multiple distinct sampled frames
# ---------------------------------------------------------------------------
def test_fragment_filter_keeps_real_students_only():
    svc = MLInferenceService.__new__(MLInferenceService)
    real = _hist(1, 0, 30, [500, 200, 800, 700])       # 30 distinct frames
    same_frame_dupes = {  # 10 observations but ALL at one frame (YOLO dupes)
        2: [{"frame": 120, "timestamp": 4.0, "bbox": [100, 200, 400, 700], "confidence": 0.5}
            for _ in range(10)],
    }
    two_frames = _hist(3, 0, 2, [900, 200, 1100, 700])  # only 2 distinct frames
    kept = svc._filter_fragment_tracks({1: real["bounding_box_history"], **same_frame_dupes, 3: two_frames["bounding_box_history"]})
    assert 1 in kept and 2 not in kept and 3 not in kept
