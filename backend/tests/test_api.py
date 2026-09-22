import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert "needle3" in body
    assert "main_finetuned" in body["needle3"]


def test_api_sample_prompts():
    res = client.get("/api/sample-prompts")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 4
    assert data[0]["id"] == "varanasi-weaver"


def test_api_extract_artisan_endpoint():
    """MAIN onboarding: pure-neural 5-field contract (no fallback enrichment)."""
    prompt = "Namaste, I am Ramesh Gowda, lacquerware artisan from Channapatna Craft Cluster, Karnataka. Phone 9123456780, pin 4321. Pehchan PEH-KA-44912, TRIFED TRIFED-KA-CHN-8821."
    res = client.post("/api/extract-artisan", json={"text": prompt})
    assert res.status_code == 200
    data = res.json()
    assert data["form_data"]["name"] == "Ramesh Gowda"
    assert data["form_data"]["phone"] == "9123456780"
    assert data["form_data"]["pin"] == "4321"
    assert data["form_data"]["pehchan_id"] == "PEH-KA-44912"
    assert data["form_data"]["trifed_id"] == "TRIFED-KA-CHN-8821"
    # Pure-neural: regional enrichment stays None
    assert data["form_data"]["cluster_name"] is None
    assert data["form_data"]["state"] is None
    assert data["quick_form_data"]["name"] == "Ramesh Gowda"
    assert data["quick_form_data"]["phone"] == "9123456780"
    assert data["quick_form_data"]["pehchan_id"] == "PEH-KA-44912"
    assert "field_metadata" in data
    assert set(data["field_metadata"].keys()) <= {"name", "phone", "pin", "pehchan_id", "trifed_id"}
    assert data["processing_time_ms"] > 0


def test_api_extract_quick_artisan_endpoint():
    # Pure-neural quick mode: omit model_name so the neural path runs (fast/heuristic returns empty).
    prompt = "I am Parth, contact me at 9038749127, pehchan ID PEH-GJ-71231"
    res = client.post("/api/extract-quick-artisan", json={"text": prompt})
    assert res.status_code == 200
    data = res.json()
    assert data["quick_form_data"]["name"] == "Parth"
    assert data["quick_form_data"]["phone"] == "9038749127"
    assert data["quick_form_data"]["pehchan_id"] == "PEH-GJ-71231"
    assert "name" in data["field_metadata"]
    assert "phone" in data["field_metadata"]
    assert "pehchan_id" in data["field_metadata"]
    # Only 3 fields in quick mode
    assert len(data["field_metadata"]) == 3


def test_api_submit_quick_onboarding():
    quick_payload = {
        "name": "Gauri Devi",
        "phone": "9876543210",
        "pehchan_id": "PEH-IND-88320"
    }
    res = client.post("/api/submit-quick-onboarding", json=quick_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["verification_status"] == "verified"
    assert data["profile"]["name"] == "Gauri Devi"


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
    assert list_res.json()["count"] >= 2
