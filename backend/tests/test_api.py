import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_sample_prompts():
    res = client.get("/api/sample-prompts")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 4
    assert data[0]["id"] == "varanasi-weaver"

def test_api_extract_artisan_endpoint():
    prompt = "Namaste, I am Ramesh Gowda, lacquerware artisan from Channapatna Craft Cluster, Karnataka. Phone 9123456780, pin 4321. Pehchan PEH-KA-44912."
    res = client.post("/api/extract-artisan", json={"text": prompt})
    assert res.status_code == 200
    data = res.json()
    assert data["form_data"]["name"] == "Ramesh Gowda"
    assert data["form_data"]["phone"] == "9123456780"
    assert data["form_data"]["cluster_name"] == "Channapatna Craft Cluster"
    assert "field_metadata" in data
    assert data["processing_time_ms"] > 0

def test_api_submit_onboarding_and_retrieve():
    form_payload = {
        "name": "Sunita Jha",
        "phone": "9845123456",
        "pin": "5678",
        "primary_dialect": "maithili",
        "craft_category": "Folk Art & Painting",
        "state": "Bihar",
        "district": "Madhubani",
        "cluster_name": "Mithila Painting Cluster",
        "pehchan_id": "PEH-BR-99012",
        "trifed_id": "TRIFED-BR-MTH-3021",
        "gi_certified": True,
        "gi_tag_name": "Madhubani Paintings (GI Reg #105)",
        "years_experience": 15
    }
    res = client.post("/api/submit-onboarding", json=form_payload)
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["success"] is True
    assert res_data["verification_status"] == "verified"

    # List submissions
    list_res = client.get("/api/submissions")
    assert list_res.status_code == 200
    assert list_res.json()["count"] >= 1
