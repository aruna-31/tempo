from fastapi.testclient import TestClient


def test_subject_crud_and_isolation(
    client: TestClient, auth_headers_faculty_a, auth_headers_faculty_b
):
    # Faculty A creates a subject
    create_res = client.post(
        "/api/v1/subjects/",
        headers=auth_headers_faculty_a,
        json={
            "code": "23CS3101",
            "name": "Distributed Systems",
            "description": "Core CS Subject"
        }
    )
    assert create_res.status_code == 201
    subject_id = create_res.json()["id"]

    # Faculty A can view their subject
    get_res_a = client.get(f"/api/v1/subjects/{subject_id}", headers=auth_headers_faculty_a)
    assert get_res_a.status_code == 200
    assert get_res_a.json()["code"] == "23CS3101"

    # Faculty B CANNOT view Faculty A's subject (Tenant Isolation)
    get_res_b = client.get(f"/api/v1/subjects/{subject_id}", headers=auth_headers_faculty_b)
    assert get_res_b.status_code == 404

    # Faculty B CANNOT update Faculty A's subject
    update_res_b = client.put(
        f"/api/v1/subjects/{subject_id}",
        headers=auth_headers_faculty_b,
        json={"name": "Hacked Subject"}
    )
    assert update_res_b.status_code == 404

    # Faculty A updates their subject
    update_res_a = client.put(
        f"/api/v1/subjects/{subject_id}",
        headers=auth_headers_faculty_a,
        json={"name": "Advanced Distributed Systems"}
    )
    assert update_res_a.status_code == 200
    assert update_res_a.json()["name"] == "Advanced Distributed Systems"


def test_section_and_student_workflow_with_scoping(
    client: TestClient, auth_headers_faculty_a, auth_headers_faculty_b
):
    # Faculty A creates subject
    sub_res = client.post(
        "/api/v1/subjects/",
        headers=auth_headers_faculty_a,
        json={"code": "23CS2202", "name": "Computer Architecture"}
    )
    subject_id = sub_res.json()["id"]

    # Faculty A creates section
    sec_res = client.post(
        "/api/v1/sections/",
        headers=auth_headers_faculty_a,
        json={
            "subject_id": subject_id,
            "name": "Section A",
            "academic_year": "2025-2026",
            "semester": "Semester 5"
        }
    )
    assert sec_res.status_code == 201
    section_id = sec_res.json()["id"]

    # Faculty B tries to create a section under Faculty A's subject -> 404
    sec_res_b = client.post(
        "/api/v1/sections/",
        headers=auth_headers_faculty_b,
        json={
            "subject_id": subject_id,
            "name": "Hacked Section",
            "academic_year": "2025-2026",
            "semester": "Semester 5"
        }
    )
    assert sec_res_b.status_code == 404

    # Faculty A enrolls single student
    st_res = client.post(
        "/api/v1/students/",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "roll_number": "2300030001",
            "name": "Aarav Kumar",
            "email": "aarav@klu.ac.in"
        }
    )
    assert st_res.status_code == 201
    student_id = st_res.json()["id"]

    # Faculty A bulk enrolls students
    bulk_res = client.post(
        "/api/v1/students/bulk",
        headers=auth_headers_faculty_a,
        json={
            "section_id": section_id,
            "students": [
                {"roll_number": "2300030002", "name": "Bhavya Reddy"},
                {"roll_number": "2300030003", "name": "Chaitanya Sai"}
            ]
        }
    )
    assert bulk_res.status_code == 201
    assert len(bulk_res.json()) == 2

    # Faculty A lists students in section
    list_st_a = client.get(f"/api/v1/students/section/{section_id}", headers=auth_headers_faculty_a)
    assert list_st_a.status_code == 200
    assert len(list_st_a.json()) == 3

    # Faculty B cannot view students of Faculty A's section
    list_st_b = client.get(f"/api/v1/students/section/{section_id}", headers=auth_headers_faculty_b)
    assert list_st_b.status_code == 404
