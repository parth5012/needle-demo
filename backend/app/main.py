import os
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    SamplePrompt,
    ArtisanOnboardingForm,
    QuickArtisanOnboardingForm,
)
from app.needle_engine import (
    Needle3InferenceEngine,
    Needle3QuickInferenceEngine,
    NEEDLE3_AVAILABLE,
    _resolve_main_weights,
)
from app.sample_data import SAMPLE_PROMPTS

app = FastAPI(
    title="Needle 3 Demo API",
    description="Natural language text-to-form filling for Artify Bharat Artisan Onboarding using Needle 3 Inference Engine",
    version="1.0.0",
)

# Enable CORS for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = Needle3InferenceEngine()
quick_engine = Needle3QuickInferenceEngine()

# In-memory storage for saved onboarding submissions
ONBOARDING_REGISTRY: List[Dict[str, Any]] = []

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Needle 3 Text-to-Form Engine",
        "docs_url": "/docs",
        "supported_schemas": ["ArtifyBharatArtisanOnboarding"],
    }

@app.get("/api/health")
def health():
    """Reports service health plus live Needle 3 neural engine status.

    `needle3.available` is True only when the `cactus-needle` package imports;
    `weights_cached` / `engine_cached` confirm the downloaded `needle3.cact`
    weights and native engine library are present. When `available` is False
    (or a request uses `model_name: needle-3-heuristic/fast`), inference
    silently degrades to the deterministic regex fallback — check
    `model_used` and per-field `source_segment` in extraction responses.
    """
    weights_cached: bool = False
    engine_cached: bool = False
    main_weights: str | None = None
    try:
        main_weights = _resolve_main_weights()
    except Exception:
        main_weights = None
    try:
        from needle.agent import fetch as _fetch

        weights_path = os.path.join(_fetch.cache_dir(3), _fetch.base_weights(3))
        weights_cached = os.path.exists(weights_path)
        lib_path = os.path.join(_fetch.cache_dir(3), _fetch._lib_name())
        engine_cached = os.path.exists(lib_path)
    except Exception:
        pass
    return {
        "status": "healthy",
        "engine": "needle-3-local",
        "version": "1.0.0",
        "needle3": {
            "available": NEEDLE3_AVAILABLE,
            "weights_cached": weights_cached,
            "engine_cached": engine_cached,
            "neural_by_default": True,
            "main_weights": main_weights,
            "main_finetuned": main_weights is not None,
        },
    }

@app.get("/api/sample-prompts", response_model=List[SamplePrompt])
def get_sample_prompts():
    """Returns sample artisan stories in multiple dialects for quick testing."""
    return SAMPLE_PROMPTS

@app.post("/api/extract-artisan", response_model=ExtractionResponse)
def extract_artisan_form(req: ExtractionRequest):
    """
    Extracts structured artisan onboarding information from freeform text / voice transcripts.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        response = engine.extract(req)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference engine error: {str(e)}")

@app.post("/api/extract-quick-artisan", response_model=ExtractionResponse)
def extract_quick_artisan_form(req: ExtractionRequest):
    """
    Extracts minimal 3-field artisan onboarding information (Name, Phone, Pehchan ID)
    using the lightweight dedicated Needle 3 Quick Engine.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        req.quick_mode = True
        response = quick_engine.extract(req)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick engine error: {str(e)}")

@app.post("/api/submit-onboarding")
def submit_onboarding(form: ArtisanOnboardingForm):
    """
    Submits the validated and reviewed artisan onboarding form.
    """
    if not form.name or not form.phone:
        raise HTTPException(status_code=422, detail="Artisan Name and Phone are required for onboarding.")

    submission = {
        "artisan_id": str(uuid.uuid4()),
        "data": form.model_dump(),
        "status": "registered",
        "verification_status": "verified" if (form.pehchan_id or form.trifed_id) else "pending_verification"
    }
    ONBOARDING_REGISTRY.append(submission)
    return {
        "success": True,
        "message": "Artisan onboarding profile successfully created in Artify Bharat registry.",
        "artisan_id": submission["artisan_id"],
        "verification_status": submission["verification_status"],
        "profile": submission["data"]
    }

@app.post("/api/submit-quick-onboarding")
def submit_quick_onboarding(form: QuickArtisanOnboardingForm):
    """
    Submits minimal artisan onboarding (Name, Phone, Pehchan ID).
    """
    if not form.name or not form.phone:
        raise HTTPException(status_code=422, detail="Artisan Name and Phone are required.")

    submission = {
        "artisan_id": str(uuid.uuid4()),
        "data": form.model_dump(),
        "type": "quick_onboarding",
        "status": "registered",
        "verification_status": "verified" if form.pehchan_id else "pending_verification"
    }
    ONBOARDING_REGISTRY.append(submission)
    return {
        "success": True,
        "message": "Quick artisan onboarding profile created.",
        "artisan_id": submission["artisan_id"],
        "verification_status": submission["verification_status"],
        "profile": submission["data"]
    }

@app.get("/api/submissions")
def list_submissions():
    """Lists submitted artisan onboarding profiles."""
    return {"count": len(ONBOARDING_REGISTRY), "submissions": ONBOARDING_REGISTRY}
