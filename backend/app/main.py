import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    SamplePrompt,
    ArtisanOnboardingForm,
)
from app.needle_engine import Needle3InferenceEngine
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
    return {"status": "healthy", "engine": "needle-3-local", "version": "1.0.0"}

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

@app.get("/api/submissions")
def list_submissions():
    """Lists submitted artisan onboarding profiles."""
    return {"count": len(ONBOARDING_REGISTRY), "submissions": ONBOARDING_REGISTRY}
