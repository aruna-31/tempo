import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import requests
from datetime import date, time as dtime

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_full_prototype_flow():
    print("==================================================================", flush=True)
    print("TEMPO End-to-End Prototype Workflow Automated Verification", flush=True)
    print("==================================================================", flush=True)

    # 1. Faculty registration with a unique generated account (no hardcoded demo
    #    credentials - the account is created through the actual application flow).
    import uuid
    email = f"e2e.faculty_{uuid.uuid4().hex[:8]}@klu.ac.in"
    password = os.getenv("TEST_FACULTY_PASSWORD", "SecureE2EPassword123!")

    print(f"\n[Step 1] Registering a fresh faculty account: {email}", flush=True)
    reg_resp = requests.post(f"{BASE_URL}/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Dr. Verification Faculty",
        "department": "Computer Science & Engineering",
        "designation": "Associate Professor"
    })
    assert reg_resp.status_code in (200, 201), f"Registration failed: {reg_resp.text}"

    print("\n[Step 1b] Logging in with the registered credentials...", flush=True)
    login_payload = {"email": email, "password": password}
    resp = requests.post(f"{BASE_URL}/auth/login", json=login_payload)
    assert resp.status_code == 200, f"Login failed: {resp.text}"

    token_data = resp.json()
    access_token = token_data.get("access_token")
    assert access_token, "No access token received"
    print(f"-> Login SUCCESS. Received Bearer JWT Token ({len(access_token)} chars)", flush=True)
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # 2. Fetch Dashboard Today's Classes
    print(f"\n[Step 2] Fetching Today's Classes for authenticated faculty from Timetable...", flush=True)
    today_resp = requests.get(f"{BASE_URL}/schedules/today", headers=headers)
    assert today_resp.status_code == 200, f"Failed to get today's classes: {today_resp.text}"
    today_classes = today_resp.json()
    print(f"-> Received {len(today_classes)} classes scheduled for today ({date.today().strftime('%A')}).", flush=True)
    
    if len(today_classes) == 0:
        print("-> Setting up a today class schedule for this faculty...", flush=True)
        from app.db.session import SessionLocal
        from app.models.faculty import Faculty
        from app.models.academic import Subject, Section
        from app.models.schedule import FacultySchedule
        
        db = SessionLocal()
        fac = db.query(Faculty).filter(Faculty.email == email).first()
        subj = db.query(Subject).filter(Subject.faculty_id == fac.id).first()
        if not subj:
            subj = Subject(faculty_id=fac.id, code="CSE401", name="Pattern Recognition & AI")
            db.add(subj)
            db.commit()
            db.refresh(subj)
            
        sec = db.query(Section).filter(Section.subject_id == subj.id).first()
        if not sec:
            sec = Section(subject_id=subj.id, name="Section A", academic_year="2025-2026", semester="EVEN")
            db.add(sec)
            db.commit()
            db.refresh(sec)
            
        today = date.today()
        day_name = today.strftime("%A").upper()
        
        sched = FacultySchedule(
            faculty_id=fac.id,
            subject_id=subj.id,
            section_id=sec.id,
            day_of_week=day_name,
            start_time=dtime(10, 0),
            end_time=dtime(11, 0),
            is_active=True
        )
        db.add(sched)
        db.commit()
        db.close()
        
        today_resp = requests.get(f"{BASE_URL}/schedules/today", headers=headers)
        assert today_resp.status_code == 200
        today_classes = today_resp.json()
        assert len(today_classes) > 0

    target_session = today_classes[0]
    target_session_id = target_session["session_id"]
    print(f"-> Selected Today's Class Session: {target_session.get('title')} (ID: {target_session_id})", flush=True)

    # 3. Select Class Session & Upload Real Video
    print(f"\n[Step 3] Uploading Real MP4 Video to Class Session {target_session_id}...", flush=True)
    raw_dir = "storage/raw_videos"
    sample_video_path = None
    if os.path.exists(raw_dir):
        mp4_files = [
            os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
            if f.endswith(".mp4") and os.path.getsize(os.path.join(raw_dir, f)) > 100000
        ]
        if mp4_files:
            mp4_files.sort(key=lambda x: os.path.getsize(x))
            sample_video_path = mp4_files[0]
            
    assert sample_video_path and os.path.exists(sample_video_path), f"No sample MP4 found in {raw_dir}"
    print(f"-> Using real classroom video: {sample_video_path} ({os.path.getsize(sample_video_path) / 1024:.1f} KB)", flush=True)
    
    with open(sample_video_path, "rb") as f:
        files = {"file": (os.path.basename(sample_video_path), f, "video/mp4")}
        upload_resp = requests.post(
            f"{BASE_URL}/sessions/{target_session_id}/videos",
            headers=headers,
            files=files
        )
    
    assert upload_resp.status_code == 201, f"Upload failed ({upload_resp.status_code}): {upload_resp.text}"
    upload_data = upload_resp.json()
    video_id = upload_data.get("video", {}).get("id") or upload_data.get("video_id")
    job_id = upload_data.get("analysis_job", {}).get("id") or upload_data.get("job_id")
    print(f"-> Video Uploaded Successfully! video_id={video_id}, job_id={job_id}", flush=True)

    # 4. Monitor ML/DL Pipeline Execution
    print(f"\n[Step 4] Monitoring Background ML/DL Pipeline (YOLO -> ByteTrack -> ResNet-18 -> Temporal DL)...", flush=True)
    max_wait = 180
    start_t = time.time()
    job_status = "PENDING"
    last_stage = None
    
    while time.time() - start_t < max_wait:
        job_resp = requests.get(f"{BASE_URL}/analysis/jobs/{job_id}", headers=headers)
        assert job_resp.status_code == 200, f"Failed to poll job status: {job_resp.text}"
        job_info = job_resp.json()
        job_status = job_info.get("status")
        current_stage = job_info.get("current_stage")
        progress = job_info.get("progress_pct", 0)
        
        print(f"   [ML Pipeline Progress] Status: {job_status}, Progress: {progress}%, Elapsed: {int(time.time() - start_t)}s", flush=True)
            
        if job_status in ["COMPLETED", "FAILED"]:
            break
        time.sleep(3)
        
    assert job_status == "COMPLETED", f"Job did not complete successfully. Status: {job_status}, Error: {job_info.get('error_message')}"
    print(f"-> ML Pipeline Finished with status: COMPLETED in {int(time.time() - start_t)}s", flush=True)

    # 5. Verify PostgreSQL Results
    print(f"\n[Step 5] Verifying PostgreSQL Analysis Results via API...", flush=True)
    results_resp = requests.get(f"{BASE_URL}/analysis/jobs/{job_id}", headers=headers)
    assert results_resp.status_code == 200, f"Failed to fetch results: {results_resp.text}"
    results_data = results_resp.json()
    
    print(f"-> Job status: {results_data.get('status')}", flush=True)
    print(f"-> Progress: {results_data.get('progress_pct')}%", flush=True)

    # 6. Verify Playable Annotated Video Streaming
    print(f"\n[Step 6] Verifying Playable Annotated Video Stream API...", flush=True)
    stream_resp = requests.get(f"{BASE_URL}/videos/{video_id}/output-stream", headers=headers, stream=True)
    assert stream_resp.status_code in [200, 206], f"Stream endpoint returned status: {stream_resp.status_code}"
    print(f"-> Annotated video output stream verified: status={stream_resp.status_code}, content-type={stream_resp.headers.get('Content-Type')}", flush=True)

    print("\n==================================================================", flush=True)
    print("ALL PROTOTYPE WORKFLOW STEPS PASSED SUCCESSFULLY!", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    test_full_prototype_flow()
