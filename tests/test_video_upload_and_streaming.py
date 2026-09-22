import io
import os
import pytest
from fastapi.testclient import TestClient


def create_sample_session(client: TestClient, auth_headers: dict) -> str:
    sub_res = client.post(
        "/api/v1/subjects/",
        headers=auth_headers,
        json={"code": "23CS9999", "name": "Video Analytics Testing"}
    )
    subject_id = sub_res.json()["id"]

    sec_res = client.post(
        "/api/v1/sections/",
        headers=auth_headers,
        json={
            "subject_id": subject_id,
            "name": "Sec-V1",
            "academic_year": "2025-2026",
            "semester": "Semester 5"
        }
    )
    section_id = sec_res.json()["id"]

    sess_res = client.post(
        "/api/v1/sessions/",
        headers=auth_headers,
        json={
            "section_id": section_id,
            "title": "Lab 01 Video Recording",
            "session_date": "2026-09-18",
            "start_time": "14:00:00",
            "end_time": "16:00:00"
        }
    )
    return sess_res.json()["id"]


def test_video_upload_valid_formats_and_auto_job_creation(
    client: TestClient, auth_headers_faculty_a
):
    session_id = create_sample_session(client, auth_headers_faculty_a)

    for ext, mime in [(".mp4", "video/mp4"), (".mov", "video/quicktime"), (".webm", "video/webm")]:
        fake_video_content = b"FAKE_VIDEO_BINARY_DATA_HEADER_" + b"A" * 1024
        files = {
            "file": (f"classroom_cam_{ext[1:]}{ext}", io.BytesIO(fake_video_content), mime)
        }
        data = {
            "session_id": session_id,
            "duration_seconds": "120.5"
        }

        response = client.post(
            "/api/v1/videos/upload",
            headers=auth_headers_faculty_a,
            data=data,
            files=files
        )

        assert response.status_code == 201, f"Failed for {ext}: {response.text}"
        payload = response.json()

        # Check Video payload
        assert "video" in payload
        assert payload["video"]["session_id"] == session_id
        assert payload["video"]["duration_seconds"] == 120.5
        assert payload["video"]["file_size_bytes"] == len(fake_video_content)

        # Check automatically created AnalysisJob
        assert "analysis_job" in payload
        job = payload["analysis_job"]
        assert job["video_id"] == payload["video"]["id"]
        assert job["status"] in ("PENDING", "PROCESSING", "QUEUED", "COMPLETED")
        assert "target_behaviours" in job["config_json"]


def test_video_upload_invalid_extensions_rejected(
    client: TestClient, auth_headers_faculty_a
):
    session_id = create_sample_session(client, auth_headers_faculty_a)

    disallowed = [
        ("malicious.exe", b"binary", "application/x-msdownload"),
        ("notes.pdf", b"%PDF-1.4...", "application/pdf"),
        ("script.py", b"print('hello')", "text/x-python"),
        ("document.docx", b"PK...", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    ]

    for filename, content, mime in disallowed:
        files = {"file": (filename, io.BytesIO(content), mime)}
        response = client.post(
            "/api/v1/videos/upload",
            headers=auth_headers_faculty_a,
            data={"session_id": session_id},
            files=files
        )
        assert response.status_code == 400
        assert "Unsupported video format" in response.json()["detail"]


def test_video_ownership_and_scoping_isolation(
    client: TestClient, auth_headers_faculty_a, auth_headers_faculty_b
):
    session_a = create_sample_session(client, auth_headers_faculty_a)

    # Faculty B attempts to upload video to Faculty A's session -> 404
    files = {"file": ("intrusion.mp4", io.BytesIO(b"data"), "video/mp4")}
    unauth_upload = client.post(
        "/api/v1/videos/upload",
        headers=auth_headers_faculty_b,
        data={"session_id": session_a},
        files=files
    )
    assert unauth_upload.status_code == 404

    # Faculty A uploads video
    upload_res = client.post(
        "/api/v1/videos/upload",
        headers=auth_headers_faculty_a,
        data={"session_id": session_a},
        files=files
    )
    assert upload_res.status_code == 201
    video_id = upload_res.json()["video"]["id"]

    # Faculty B cannot get video metadata
    get_meta_b = client.get(f"/api/v1/videos/{video_id}", headers=auth_headers_faculty_b)
    assert get_meta_b.status_code == 404

    # Faculty B cannot check video status
    get_status_b = client.get(f"/api/v1/videos/{video_id}/status", headers=auth_headers_faculty_b)
    assert get_status_b.status_code == 404

    # Faculty B cannot stream the video
    stream_b = client.get(f"/api/v1/videos/{video_id}/stream", headers=auth_headers_faculty_b)
    assert stream_b.status_code == 404


def test_http_range_video_streaming(
    client: TestClient, auth_headers_faculty_a
):
    session_id = create_sample_session(client, auth_headers_faculty_a)

    # 100 bytes of deterministic test data: 0, 1, 2, ... 99
    test_binary = bytes(range(100))
    files = {"file": ("test_streaming.mp4", io.BytesIO(test_binary), "video/mp4")}
    upload_res = client.post(
        "/api/v1/videos/upload",
        headers=auth_headers_faculty_a,
        data={"session_id": session_id},
        files=files
    )
    video_id = upload_res.json()["video"]["id"]

    # 1. Full Stream (no Range header) -> 200 OK
    res_full = client.get(
        f"/api/v1/videos/{video_id}/stream",
        headers=auth_headers_faculty_a
    )
    assert res_full.status_code == 200
    assert res_full.headers["Accept-Ranges"] == "bytes"
    assert res_full.headers["Content-Length"] == "100"
    assert res_full.content == test_binary

    # 2. Byte Range Request (bytes=0-49) -> 206 Partial Content
    res_part1 = client.get(
        f"/api/v1/videos/{video_id}/stream",
        headers={**auth_headers_faculty_a, "Range": "bytes=0-49"}
    )
    assert res_part1.status_code == 206
    assert res_part1.headers["Content-Range"] == "bytes 0-49/100"
    assert res_part1.headers["Content-Length"] == "50"
    assert res_part1.content == test_binary[0:50]

    # 3. Open-ended Byte Range Request (bytes=50-) -> 206 Partial Content
    res_part2 = client.get(
        f"/api/v1/videos/{video_id}/stream",
        headers={**auth_headers_faculty_a, "Range": "bytes=50-"}
    )
    assert res_part2.status_code == 206
    assert res_part2.headers["Content-Range"] == "bytes 50-99/100"
    assert res_part2.headers["Content-Length"] == "50"
    assert res_part2.content == test_binary[50:100]

    # 4. Middle Byte Range Request (bytes=20-30) -> 206 Partial Content
    res_part3 = client.get(
        f"/api/v1/videos/{video_id}/stream",
        headers={**auth_headers_faculty_a, "Range": "bytes=20-30"}
    )
    assert res_part3.status_code == 206
    assert res_part3.headers["Content-Range"] == "bytes 20-30/100"
    assert res_part3.headers["Content-Length"] == "11"
    assert res_part3.content == test_binary[20:31]


def test_video_status_endpoint_and_output_video_stream(
    client: TestClient, auth_headers_faculty_a
):
    session_id = create_sample_session(client, auth_headers_faculty_a)
    fake_content = b"SAMPLE_VIDEO_CONTENT_FOR_STATUS"
    files = {"file": ("demo.mp4", io.BytesIO(fake_content), "video/mp4")}

    upload_res = client.post(
        "/api/v1/videos/upload",
        headers=auth_headers_faculty_a,
        data={"session_id": session_id},
        files=files
    )
    video_id = upload_res.json()["video"]["id"]
    job_id = upload_res.json()["analysis_job"]["id"]

    # Check status endpoint
    status_res = client.get(
        f"/api/v1/videos/{video_id}/status",
        headers=auth_headers_faculty_a
    )
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["video_id"] == video_id
    assert status_data["latest_analysis_job_id"] == job_id
    assert status_data["stream_url"] == f"/api/v1/videos/{video_id}/stream"

    # Attempt to stream output video before completion -> 404
    out_stream_res = client.get(
        f"/api/v1/videos/{video_id}/output-stream",
        headers=auth_headers_faculty_a
    )
    assert out_stream_res.status_code == 404


def test_five_observable_behaviour_classes_validation(
    client: TestClient, auth_headers_faculty_a
):
    session_id = create_sample_session(client, auth_headers_faculty_a)
    files = {"file": ("behav.mp4", io.BytesIO(b"data"), "video/mp4")}
    upload_res = client.post(
        "/api/v1/videos/upload",
        headers=auth_headers_faculty_a,
        data={"session_id": session_id},
        files=files
    )
    job_id = upload_res.json()["analysis_job"]["id"]

    # Ingest the exact 5 observable behaviours -> Should succeed
    five_behaviours = [
        {"job_id": job_id, "track_id": 1, "frame_number": 0, "timestamp_seconds": 0.0, "behaviour_type": "Looking_Toward_Instruction", "confidence": 0.95},
        {"job_id": job_id, "track_id": 1, "frame_number": 30, "timestamp_seconds": 1.0, "behaviour_type": "Reading", "confidence": 0.88},
        {"job_id": job_id, "track_id": 1, "frame_number": 60, "timestamp_seconds": 2.0, "behaviour_type": "Writing", "confidence": 0.91},
        {"job_id": job_id, "track_id": 1, "frame_number": 90, "timestamp_seconds": 3.0, "behaviour_type": "Peer_Interaction", "confidence": 0.85},
        {"job_id": job_id, "track_id": 1, "frame_number": 120, "timestamp_seconds": 4.0, "behaviour_type": "Looking_Away", "confidence": 0.79}
    ]
    res_valid = client.post(
        f"/api/v1/analysis/jobs/{job_id}/behaviours",
        headers=auth_headers_faculty_a,
        json={"job_id": job_id, "results": five_behaviours}
    )
    assert res_valid.status_code == 201
    assert len(res_valid.json()) == 5

    # Ingest invalid behaviour type -> Should fail with 422
    invalid_behaviour = [
        {"job_id": job_id, "track_id": 1, "frame_number": 150, "timestamp_seconds": 5.0, "behaviour_type": "UNSUPPORTED_ACTION", "confidence": 0.5}
    ]
    res_invalid = client.post(
        f"/api/v1/analysis/jobs/{job_id}/behaviours",
        headers=auth_headers_faculty_a,
        json={"job_id": job_id, "results": invalid_behaviour}
    )
    assert res_invalid.status_code == 422
