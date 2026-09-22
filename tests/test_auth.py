from fastapi.testclient import TestClient
from app.core.security import create_refresh_token


def test_register_valid_klu_email(client: TestClient):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "faculty.cse@klu.ac.in",
            "password": "SecurePassword123",
            "full_name": "CSE Faculty",
            "department": "Computer Science",
            "designation": "Assistant Professor"
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "faculty.cse@klu.ac.in"
    assert data["full_name"] == "CSE Faculty"
    assert "id" in data
    assert "hashed_password" not in data


def test_register_invalid_email_domain_rejected(client: TestClient):
    invalid_emails = [
        "faculty@gmail.com",
        "faculty@klu.edu",
        "faculty@outlook.com",
        "faculty@klu.ac.org"
    ]
    for email in invalid_emails:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": email,
                "password": "SecurePassword123",
                "full_name": "Test Faculty",
                "department": "Computer Science"
            }
        )
        assert response.status_code == 422, f"Expected 422 for non-KLU email: {email}"


def test_register_short_password_rejected(client: TestClient):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "short.pwd@klu.ac.in",
            "password": "short",
            "full_name": "Test Faculty",
            "department": "Computer Science"
        }
    )
    assert response.status_code == 422


def test_register_duplicate_email_rejected(client: TestClient, faculty_a):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": faculty_a.email,
            "password": "AnotherPassword123",
            "full_name": "Duplicate Faculty",
            "department": "CSE"
        }
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_login_success(client: TestClient, faculty_a):
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "prof.sharma@klu.ac.in",
            "password": "SecretPassword123!"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_password(client: TestClient, faculty_a):
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "prof.sharma@klu.ac.in",
            "password": "WrongPassword!"
        }
    )
    assert response.status_code == 401


def test_login_invalid_domain_rejected(client: TestClient):
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "hacker@gmail.com",
            "password": "SomePassword123!"
        }
    )
    assert response.status_code == 422


def test_refresh_token(client: TestClient, faculty_a):
    valid_refresh = create_refresh_token(subject=str(faculty_a.id))
    response = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": valid_refresh}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data


def test_get_current_profile_me(client: TestClient, auth_headers_faculty_a, faculty_a):
    response = client.get("/api/v1/auth/me", headers=auth_headers_faculty_a)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(faculty_a.id)
    assert data["email"] == faculty_a.email


def test_get_me_unauthorized(client: TestClient):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_update_profile(client: TestClient, auth_headers_faculty_a):
    response = client.put(
        "/api/v1/auth/me",
        headers=auth_headers_faculty_a,
        json={
            "full_name": "Dr. Sharma Updated",
            "designation": "Head of Department"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["full_name"] == "Dr. Sharma Updated"
    assert data["designation"] == "Head of Department"
