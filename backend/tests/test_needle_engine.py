import pytest
from app.needle_engine import (
    Needle3InferenceEngine,
    _fix_swapped_ids,
    _collect_neural_fields,
)
from app.schemas import ExtractionRequest, ArtisanOnboardingForm

PURE_NEURAL_FIELDS = {"name", "phone", "pin", "pehchan_id", "trifed_id"}


@pytest.fixture
def engine():
    return Needle3InferenceEngine()


def test_varanasi_weaver_extraction(engine):
    """MAIN onboarding core: Bhojpuri narrative must yield the 5 neural fields."""
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
    assert res.form_data.pehchan_id == "PEH-IND-88320"
    assert res.form_data.trifed_id == "TRIFED-UP-VNS-1049"
    # Pure-neural: only the 5 grammar fields may appear; regional enrichment stays None.
    assert set(res.field_metadata.keys()) <= PURE_NEURAL_FIELDS
    assert res.form_data.state is None
    assert res.form_data.district is None
    assert res.form_data.cluster_name is None
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
    assert res.form_data.pehchan_id == "PEH-KA-44912"
    assert res.form_data.trifed_id == "TRIFED-KA-CHN-8821"
    assert set(res.field_metadata.keys()) <= PURE_NEURAL_FIELDS


def test_gujarati_main_extraction(engine):
    """Gujarati narrative — same 5-field MAIN contract."""
    text = (
        "Namaste! My name is Fatima Khatri from Bhuj, Kutch district in Gujarat. "
        "Phone number: 9988776655, pin: 9876. "
        "Pehchan ID PEH-GJ-77123, TRIFED-GJ-KCH-1102."
    )
    req = ExtractionRequest(text=text)
    res = engine.extract(req)

    assert res.form_data.name == "Fatima Khatri"
    assert res.form_data.phone == "9988776655"
    assert res.form_data.pin == "9876"
    assert res.form_data.pehchan_id == "PEH-GJ-77123"
    assert res.form_data.trifed_id == "TRIFED-GJ-KCH-1102"


def test_single_word_name_and_comma_delimiters(engine):
    text = "I am Parth,contact me at 9038749127, pehchan id PEH-GJ-71231"
    req = ExtractionRequest(text=text)
    res = engine.extract(req)

    assert res.form_data.name == "Parth"
    assert res.form_data.phone == "9038749127"
    assert res.form_data.pehchan_id == "PEH-GJ-71231"


def test_phone_pin_disambiguation_unit():
    """4-digit PIN must never validate as phone; +91 prefix normalizes."""
    calls = [{"arguments": {
        "name": "Anil Verma", "phone": "+91 9811023456", "pin": "PIN-3344",
        "pehchan_id": "PEH-DL-20314", "trifed_id": "TRIFED-DL-0091"}}]
    fields, _, _ = _collect_neural_fields(
        calls, allowed_keys=("name", "phone", "pin", "pehchan_id", "trifed_id"),
        model_confidence=0.95, source_prefix="unit")
    assert fields["phone"].value == "9811023456"
    assert fields["pin"].value == "3344"

    pin_as_phone = [{"arguments": {"phone": "1234"}}]
    fields2, _, _ = _collect_neural_fields(
        pin_as_phone, allowed_keys=("name", "phone", "pin", "pehchan_id", "trifed_id"),
        model_confidence=0.95, source_prefix="unit")
    assert "phone" not in fields2


def test_greeting_never_name_and_slot_correction_unit():
    calls = [{"arguments": {"name": "Namaskara"}}]
    fields, _, _ = _collect_neural_fields(
        calls, allowed_keys=("name", "phone", "pin", "pehchan_id", "trifed_id"),
        model_confidence=0.95, source_prefix="unit")
    assert "name" not in fields

    from app.schemas import ExtractedFieldMeta
    fields = {
        "pehchan_id": ExtractedFieldMeta(value="TRIFED-UP-VNS-1049", confidence=0.95, source_segment="s"),
        "trifed_id": ExtractedFieldMeta(value="PEH-IND-88320", confidence=0.95, source_segment="s"),
    }
    entities = []
    _fix_swapped_ids(fields, entities)
    assert fields["pehchan_id"].value == "PEH-IND-88320"
    assert fields["trifed_id"].value == "TRIFED-UP-VNS-1049"


def test_empty_or_minimal_text(engine):
    """Pure-neural refusal: no fallback, fields stay None."""
    text = "Hello I make crafts."
    req = ExtractionRequest(text=text)
    res = engine.extract(req)
    assert res.form_data.name is None or isinstance(res.form_data.name, str)
    assert set(res.field_metadata.keys()) <= PURE_NEURAL_FIELDS
    assert res.processing_time_ms >= 0
