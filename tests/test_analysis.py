from fastapi.testclient import TestClient


def test_analysis_job_and_temporal_behaviour_analytics(
    client: TestClient, auth_headers_faculty_a, auth_headers_faculty_b
):
    # 1. Setup Academic hierarchy
    sub_res = client.post(
        "/api/v1/subjects/",
        headers=auth_headers_faculty_a,
        json={"code": "23CS2404", "name": "Artificial Intelligence"}
    )
    subject_id = sub_res.json()["id"]

    sec_res = client.post(
        "/api/v1/sections/",
        headers=auth_headers_faculty_a,
        json={
            "subject_id": subject_id,
            "name": "Section C",
            "academic_year": "2025-2026",
            "semester": "Semester 5"
        }
    )
    section_id = sec_res.json()["id"]

    st_res = client.post(
        "/api/v1/students/",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "roll_number": "2300030099",
            "name": "Divya Prakash"
        }
    )
    student_id = st_res.json()["id"]

    sess_res = client.post(
        "/api/v1/sessions/",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "title": "AI Unit 1: Search Algorithms",
            "session_date": "2026-09-17",
            "start_time": "11:00:00",
            "end_time": "12:30:00"
        }
    )
    session_id = sess_res.json()["id"]

    vid_res = client.post(
        "/api/v1/videos/",
        headers=auth_headers_faculty_a,
        json={
            "session_id": session_id,
            "original_filename": "ai_lecture_c_cam1.mp4",
            "file_size_bytes": 52428800,
            "file_path": "/uploads/videos/ai_cam1.mp4",
            "duration_seconds": 180.0
        }
    )
    video_id = vid_res.json()["id"]

    # 2. Create Analysis Job
    job_res = client.post(
        "/api/v1/analysis/jobs",
        headers=auth_headers_faculty_a,
        json={
            "video_id": video_id,
            "config_json": {"model": "temporal_behaviour_v1", "fps": 30}
        }
    )
    assert job_res.status_code == 201
    job_id = job_res.json()["id"]
    assert job_res.json()["status"] in ("PENDING", "PROCESSING", "QUEUED")

    # Faculty B cannot view this job
    job_res_b = client.get(f"/api/v1/analysis/jobs/{job_id}", headers=auth_headers_faculty_b)
    assert job_res_b.status_code == 404

    # 3. Update job status to PROCESSING and COMPLETED
    patch_job = client.patch(
        f"/api/v1/analysis/jobs/{job_id}/status",
        headers=auth_headers_faculty_a,
        json={"status": "PROCESSING", "progress_pct": 50}
    )
    assert patch_job.status_code == 200
    assert patch_job.json()["status"] == "PROCESSING"
    assert patch_job.json()["progress_pct"] == 50

    # 4. Save Student Track Result (Temporary per-session Track ID)
    track_res = client.post(
        f"/api/v1/analysis/jobs/{job_id}/tracks",
        headers=auth_headers_faculty_a,
        json={
            "job_id": job_id,
            "student_id": student_id,
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 1500,
            "bounding_box_history": [
                {"frame": 0, "timestamp": 0.0, "bbox": [100, 150, 200, 300]},
                {"frame": 30, "timestamp": 1.0, "bbox": [102, 152, 202, 302]}
            ]
        }
    )
    assert track_res.status_code == 201
    assert track_res.json()["track_id"] == 1

    # 5. Ingest Behaviour Results (Batch with 5 observable behaviour types)
    behaviours_payload = [
        {
            "job_id": job_id,
            "student_id": student_id,
            "track_id": 1,
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "behaviour_type": "Looking_Toward_Instruction",
            "confidence": 0.95
        },
        {
            "job_id": job_id,
            "student_id": student_id,
            "track_id": 1,
            "frame_number": 30,
            "timestamp_seconds": 1.0,
            "behaviour_type": "Reading",
            "confidence": 0.92
        },
        {
            "job_id": job_id,
            "student_id": student_id,
            "track_id": 1,
            "frame_number": 1800,
            "timestamp_seconds": 60.0,
            "behaviour_type": "Writing",
            "confidence": 0.88
        },
        {
            "job_id": job_id,
            "student_id": student_id,
            "track_id": 1,
            "frame_number": 3600,
            "timestamp_seconds": 120.0,
            "behaviour_type": "Peer_Interaction",
            "confidence": 0.91
        }
    ]
    bh_res = client.post(
        f"/api/v1/analysis/jobs/{job_id}/behaviours",
        headers=auth_headers_faculty_a,
        json={"job_id": job_id, "results": behaviours_payload}
    )
    assert bh_res.status_code == 201
    assert len(bh_res.json()) == 4

    # Complete job
    client.patch(
        f"/api/v1/analysis/jobs/{job_id}/status",
        headers=auth_headers_faculty_a,
        json={"status": "COMPLETED", "progress_pct": 100}
    )

    # 6. Retrieve Temporal Summary Analytics
    summary_res = client.get(
        f"/api/v1/analysis/jobs/{job_id}/summary?bucket_seconds=60.0",
        headers=auth_headers_faculty_a
    )
    assert summary_res.status_code == 200
    summary = summary_res.json()

    assert summary["job_id"] == job_id
    assert summary["status"] == "COMPLETED"
    assert summary["total_frames_analyzed"] == 4
    assert summary["overall_engagement_score"] == 75.0  # 3 engaged (Instruction, Reading, Writing) out of 4 total

    # Verify timeline buckets
    assert len(summary["timeline_buckets"]) == 3  # 0-60, 60-120, 120-180
    assert summary["timeline_buckets"][0]["looking_toward_instruction_count"] == 1
    assert summary["timeline_buckets"][0]["reading_count"] == 1
    assert summary["timeline_buckets"][0]["engagement_index"] == 100.0

    # Verify track metrics (per-session Track ID)
    assert len(summary["track_metrics"]) == 1
    t_metric = summary["track_metrics"][0]
    assert t_metric["track_id"] == 1
    assert t_metric["looking_toward_instruction_pct"] == 25.0
    assert t_metric["reading_pct"] == 25.0
    assert t_metric["writing_pct"] == 25.0
    assert t_metric["peer_interaction_pct"] == 25.0
