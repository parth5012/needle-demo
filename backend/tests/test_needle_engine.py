import pytest
from app.needle_engine import Needle3InferenceEngine
from app.schemas import ExtractionRequest, ArtisanOnboardingForm

@pytest.fixture
def engine():
    return Needle3InferenceEngine()

def test_varanasi_weaver_extraction(engine):
    text = (
        "Pranam! Humar naam Gauri Devi ba. Hum Varanasi Handloom Cluster me "
        "pichhle 18 years se authentic Banarasi brocade aur pure silk saree bunat baani. "
        "Contact number 9876543210 ba aur humar security pin 1234. "
        "Humaar government Pehchan ID PEH-IND-88320 ha aur TRIFED ID TRIFED-UP-VNS-1049. "
        "Banarasi Brocade GI tag craft."
    )
    req = ExtractionRequest(text=text)
    res = engine.extract(req)

    assert res.form_data.name == "Gauri Devi"
    assert res.form_data.phone == "9876543210"
    assert res.form_data.pin == "1234"
    assert res.form_data.primary_dialect == "bhojpuri"
    assert res.form_data.state == "Uttar Pradesh"
    assert res.form_data.district == "Varanasi"
    assert res.form_data.cluster_name == "Varanasi Handloom Cluster"
    assert res.form_data.pehchan_id == "PEH-IND-88320"
    assert res.form_data.trifed_id == "TRIFED-UP-VNS-1049"
    assert res.form_data.gi_certified is True
    assert res.form_data.years_experience == 18
    assert res.overall_confidence > 0.8
    assert res.processing_time_ms >= 0

def test_channapatna_toy_maker_extraction(engine):
    text = (
        "Namaskara, I am Ramesh Gowda, a traditional lacquerware artisan from Channapatna Craft Cluster, "
        "Ramanagara, Karnataka. 12 years of experience crafting Channapatna wooden toys. "
        "Phone: 9123456780, pin: 4321. Pehchan card is PEH-KA-44912, TRIFED ID is TRIFED-KA-CHN-8821. "
        "GI certified toys."
    )
    req = ExtractionRequest(text=text)
    res = engine.extract(req)

    assert res.form_data.name == "Ramesh Gowda"
    assert res.form_data.phone == "9123456780"
    assert res.form_data.pin == "4321"
    assert res.form_data.state == "Karnataka"
    assert res.form_data.district == "Ramanagara"
    assert res.form_data.craft_category == "Wooden Craft & Toys"
    assert res.form_data.pehchan_id == "PEH-KA-44912"
    assert res.form_data.years_experience == 12

def test_empty_or_minimal_text(engine):
    text = "Hello I make crafts."
    req = ExtractionRequest(text=text)
    res = engine.extract(req)
    assert res.form_data.name is None or isinstance(res.form_data.name, str)
    assert res.processing_time_ms >= 0
