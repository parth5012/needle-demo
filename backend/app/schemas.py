from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ExtractionRequest(BaseModel):
    text: str = Field(..., description="Raw unstructured natural language narrative or voice transcript of artisan")
    model_name: Optional[str] = Field("needle-3-local", description="Model backend identifier")
    strict_mode: Optional[bool] = Field(False, description="Whether to reject low-confidence fields")
    quick_mode: Optional[bool] = Field(False, description="Whether to extract only the 3 essential quick fields (name, phone, pehchan_id)")

class ExtractedFieldMeta(BaseModel):
    value: Optional[Any] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source_segment: Optional[str] = None

class QuickArtisanOnboardingForm(BaseModel):
    name: Optional[str] = Field(None, description="Full name of artisan")
    phone: Optional[str] = Field(None, description="10-digit mobile contact number")
    pehchan_id: Optional[str] = Field(None, description="Government Pehchan card identifier")

class ArtisanOnboardingForm(BaseModel):
    name: Optional[str] = Field(None, description="Full name of artisan")
    phone: Optional[str] = Field(None, description="10-digit mobile contact number")
    pin: Optional[str] = Field(None, description="4-digit security PIN if mentioned")
    primary_dialect: Optional[str] = Field("hindi", description="Artisan dialect / language")
    craft_category: Optional[str] = Field(None, description="Primary craft domain")
    state: Optional[str] = Field(None, description="State of origin")
    district: Optional[str] = Field(None, description="District / City")
    cluster_name: Optional[str] = Field(None, description="Artisan craft cluster name")
    trifed_id: Optional[str] = Field(None, description="Government TRIFED identifier")
    pehchan_id: Optional[str] = Field(None, description="Pehchan card identifier")
    gi_tag_name: Optional[str] = Field(None, description="Geographical Indication name if certified")
    gi_certified: bool = Field(False, description="Whether craft holds GI certification")
    years_experience: Optional[int] = Field(None, description="Years of artisanal experience")
    skills_summary: Optional[str] = Field(None, description="Extracted artisan backstory / summary")

class ExtractionResponse(BaseModel):
    form_data: ArtisanOnboardingForm
    quick_form_data: Optional[QuickArtisanOnboardingForm] = None
    field_metadata: Dict[str, ExtractedFieldMeta] = {}
    model_used: str = "needle-3-local"
    processing_time_ms: float = 0.0
    overall_confidence: float = 0.0
    extracted_entities: List[Dict[str, Any]] = []

class SamplePrompt(BaseModel):
    id: str
    title: str
    dialect: str
    category: str
    text: str
