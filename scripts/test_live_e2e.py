import os
import sys
import time
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_full_pipeline():
    print("=" * 70)
    print("TEMPO LIVE END-TO-END VERIFICATION SUITE")
    print("=" * 70)

    # 1. Health check
    print("\n[STEP 1] Testing /health endpoint...")
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print("Health response:", health_data)
    assert health_data.get("database") == "healthy", f"Database not healthy: {health_data}"
    print("PASS: System and PostgreSQL are healthy.")

    # 2. Register Faculty
    test_faculty_email = f"e2e.faculty_{int(time.time())}@klu.ac.in"
    password = "KLU_Faculty_Secure#2026"
    print(f"\n[STEP 2] Registering Faculty ({test_faculty_email})...")
    reg_payload = {
        "email": test_faculty_email,
        "password": password,
        "full_name": "Dr. Arun KLU",
        "department": "Computer Science & Engineering",
        "designation": "Associate Professor"
    }
    resp = requests.post(f"{BASE_URL}/api/v1/auth/register", json=reg_payload)
    assert resp.status_code in (201, 200), f"Registration failed: {resp.status_code} {resp.text}"
    faculty_data = resp.json()
    print("Registration successful:", faculty_data["email"], faculty_data["id"])
    print("PASS: Faculty registered in PostgreSQL.")

    # 2b. Test Invalid Domain Rejection
    print("\n[STEP 2b] Testing Invalid Email Domain Rejection (non-klu.ac.in)...")
    invalid_reg_payload = {
        "email": "intruder@gmail.com",
        "password": "Password123!",
        "full_name": "Intruder User",
        "department": "Unknown"
    }
    resp = requests.post(f"{BASE_URL}/api/v1/auth/register", json=invalid_reg_payload)
    assert resp.status_code in (400, 422), f"Expected 400/422 for invalid domain, got {resp.status_code}"
    print("PASS: Non-KLU domain correctly rejected.")

    # 3. Login
    print(f"\n[STEP 3] Authenticating Faculty via /api/v1/auth/login...")
    login_payload = {
        "email": test_faculty_email,
        "password": password
    }
    resp = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_payload)
    assert resp.status_code == 200, f"Login failed: {resp.status_code} {resp.text}"
    token_data = resp.json()
    access_token = token_data["access_token"]
    print("Login successful! Access token received (length: ", len(access_token), ")")
    headers = {"Authorization": f"Bearer {access_token}"}
    print("PASS: JWT Token generated successfully.")

    # 4. Verify /me
    print("\n[STEP 4] Verifying Profile via /api/v1/auth/me...")
    resp = requests.get(f"{BASE_URL}/api/v1/auth/me", headers=headers)
    assert resp.status_code == 200, f"Get /me failed: {resp.status_code} {resp.text}"
    profile = resp.json()
    print("Profile verified:", profile["full_name"], profile["department"])
    print("PASS: Authenticated /me endpoint verified.")

    # 5. Create Subject
    print("\n[STEP 5] Creating Subject...")
    subj_code = f"CS{int(time.time()) % 10000}"
    resp = requests.post(
        f"{BASE_URL}/api/v1/subjects",
        json={"code": subj_code, "name": "Deep Learning & Computer Vision"},
        headers=headers
    )
    assert resp.status_code in (200, 201), f"Create subject failed: {resp.text}"
    subject_id = resp.json()["id"]
    print(f"Subject created: {subj_code} (ID: {subject_id})")

    # 6. Create Section
    print("\n[STEP 6] Creating Section...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/sections",
        json={
            "name": "Section A",
            "subject_id": subject_id,
            "academic_year": "2026-2027",
            "semester": "Semester 1"
        },
        headers=headers
    )
    assert resp.status_code in (200, 201), f"Create section failed: {resp.text}"
    section_id = resp.json()["id"]
    print(f"Section created: Section A (ID: {section_id})")

    # 7. Create Class Session
    print("\n[STEP 7] Creating Class Session...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/sessions",
        json={
            "section_id": section_id,
            "title": "Lecture 01 - Temporal AI in Classroom",
            "session_date": "2026-09-16",
            "start_time": "09:00:00",
            "end_time": "10:00:00",
            "room_number": "C-301"
        },
        headers=headers
    )
    assert resp.status_code in (200, 201), f"Create session failed: {resp.text}"
    session_id = resp.json()["id"]
    print(f"Session created: ID {session_id}")
    print("PASS: Academic entities created and linked in PostgreSQL.")

    # 8. Upload Real Classroom Video
    video_path = os.path.abspath("storage/test_videos/classroom_real_persons.mp4")
    assert os.path.exists(video_path), f"Test video not found at {video_path}"
    print(f"\n[STEP 8] Uploading real classroom MP4 video ({video_path}, size: {os.path.getsize(video_path)} bytes)...")
    
    with open(video_path, "rb") as vf:
        files = {"file": ("classroom_real_persons.mp4", vf, "video/mp4")}
        data = {"session_id": str(session_id)}
        resp = requests.post(
            f"{BASE_URL}/api/v1/videos/upload",
            data=data,
            files=files,
            headers=headers
        )
    assert resp.status_code in (200, 201), f"Upload failed: {resp.status_code} {resp.text}"
    video_upload_data = resp.json()
    video_data = video_upload_data["video"]
    video_id = video_data["id"]
    job_id = video_upload_data["analysis_job"]["id"]
    print(f"Video uploaded successfully! Video ID: {video_id}, Analysis Job ID: {job_id}")
    print("PASS: Video upload stored and job dispatched.")

    # 9. If job_id not directly returned, create or retrieve analysis job
    if not job_id:
        print("\n[STEP 9] Dispatching AnalysisJob for Video...")
        resp = requests.post(
            f"{BASE_URL}/api/v1/analysis/jobs",
            json={"video_id": video_id},
            headers=headers
        )
        assert resp.status_code in (200, 201), f"Job dispatch failed: {resp.text}"
        job_id = resp.json()["id"]
    
    print(f"Monitoring AnalysisJob: {job_id}...")

    # 10. Poll for Job Completion
    max_wait = 120
    start_poll = time.time()
    job_completed = False
    job_status_data = {}

    while time.time() - start_poll < max_wait:
        resp = requests.get(f"{BASE_URL}/api/v1/analysis/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200, f"Poll job failed: {resp.text}"
        job_status_data = resp.json()
        status_val = job_status_data.get("status")
        progress = job_status_data.get("progress_pct", 0)
        print(f"  --> Job Status: {status_val} | Progress: {progress}%")

        if status_val == "COMPLETED":
            job_completed = True
            break
        elif status_val == "FAILED":
            raise RuntimeError(f"Job failed with error: {job_status_data.get('error_message')}")

        time.sleep(3)

    assert job_completed, f"Job did not complete within {max_wait}s. Last status: {job_status_data}"
    print("PASS: Real ML/DL pipeline execution completed successfully!")

    # 11. Verify Database Results (Tracks & Behaviour Predictions)
    print("\n[STEP 11] Fetching Analysis Results & Tracks from PostgreSQL...")
    resp = requests.get(f"{BASE_URL}/api/v1/analysis/jobs/{job_id}/results", headers=headers)
    assert resp.status_code == 200, f"Failed to get results: {resp.text}"
    results = resp.json()
    print(f"Total behaviour predictions saved: {len(results)}")
    assert len(results) > 0, "No behaviour predictions saved in database!"
    
    sample = results[0]
    print("Sample Behaviour Prediction:", {
        "track_id": sample.get("track_id"),
        "behaviour_type": sample.get("behaviour_type"),
        "confidence": sample.get("confidence"),
        "timestamp_seconds": sample.get("timestamp_seconds")
    })

    # Verify Tracks
    resp = requests.get(f"{BASE_URL}/api/v1/analysis/jobs/{job_id}/tracks", headers=headers)
    assert resp.status_code == 200, f"Failed to get tracks: {resp.text}"
    tracks = resp.json()
    print(f"Total student tracks saved: {len(tracks)}")
    assert len(tracks) > 0, "No student tracks saved in database!"
    for t in tracks:
        print(f"  Track ID {t.get('track_id')}: start_frame={t.get('start_frame')}, end_frame={t.get('end_frame')}, history_points={len(t.get('bounding_box_history', []))}")

    # Verify Summary
    resp = requests.get(f"{BASE_URL}/api/v1/analysis/jobs/{job_id}/summary", headers=headers)
    assert resp.status_code == 200, f"Failed to get summary: {resp.text}"
    summary = resp.json()
    print("Classroom Analysis Summary:", json.dumps(summary, indent=2))
    print("PASS: All ML/DL predictions and tracks persisted and verified in PostgreSQL.")

    # 12. Verify Video Output & Streaming
    print("\n[STEP 12] Verifying Annotated Video Output & Streaming API...")
    resp = requests.get(f"{BASE_URL}/api/v1/videos/{video_id}", headers=headers)
    assert resp.status_code == 200, f"Get video details failed: {resp.text}"
    video_info = resp.json()
    annotated_path = video_info.get("output_video_path")
    print(f"Annotated Video File Path: {annotated_path}")
    assert annotated_path and os.path.exists(annotated_path), f"Annotated video file does not exist at {annotated_path}"
    assert os.path.getsize(annotated_path) > 0, "Annotated video file is 0 bytes!"
    print(f"Annotated video size: {os.path.getsize(annotated_path)} bytes")

    # Test Streaming Endpoint with Range Header
    stream_headers = {"Range": "bytes=0-1024", **headers}
    resp = requests.get(f"{BASE_URL}/api/v1/videos/{video_id}/stream", headers=stream_headers)
    assert resp.status_code in (200, 206), f"Stream endpoint failed: {resp.status_code}"
    print(f"Video raw stream response: {resp.status_code}, content-range: {resp.headers.get('content-range')}")

    resp_out = requests.get(f"{BASE_URL}/api/v1/videos/{video_id}/output-stream", headers=stream_headers)
    assert resp_out.status_code in (200, 206), f"Output stream endpoint failed: {resp_out.status_code}"
    print(f"Annotated video stream response: {resp_out.status_code}, content-range: {resp_out.headers.get('content-range')}")
    print("PASS: Annotated video generated and playable streaming endpoints verified.")

    # 13. Verify Faculty Isolation
    print("\n[STEP 13] Verifying Multi-Tenant Faculty Isolation...")
    other_faculty_email = f"other.faculty_{int(time.time())}@klu.ac.in"
    reg_other = requests.post(f"{BASE_URL}/api/v1/auth/register", json={
        "email": other_faculty_email,
        "password": password,
        "full_name": "Dr. Other Faculty",
        "department": "ECE"
    })
    assert reg_other.status_code in (200, 201)
    login_other = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
        "email": other_faculty_email,
        "password": password
    })
    other_headers = {"Authorization": f"Bearer {login_other.json()['access_token']}"}

    # Attempt to access faculty 1's session and analysis job
    resp_unauth_session = requests.get(f"{BASE_URL}/api/v1/sessions/{session_id}", headers=other_headers)
    assert resp_unauth_session.status_code in (403, 404), f"Isolation failure on session: {resp_unauth_session.status_code}"

    resp_unauth_job = requests.get(f"{BASE_URL}/api/v1/analysis/jobs/{job_id}", headers=other_headers)
    assert resp_unauth_job.status_code in (403, 404), f"Isolation failure on job: {resp_unauth_job.status_code}"

    print("PASS: Multi-tenant faculty isolation strictly enforced (HTTP 403/404 on cross-faculty access).")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED WITH 100% SUCCESS")
    print("=" * 70)

if __name__ == "__main__":
    test_full_pipeline()
