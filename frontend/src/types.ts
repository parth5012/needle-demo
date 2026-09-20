export interface ArtisanOnboardingForm {
  name: string | null;
  phone: string | null;
  pin: string | null;
  primary_dialect: string;
  craft_category: string | null;
  state: string | null;
  district: string | null;
  cluster_name: string | null;
  trifed_id: string | null;
  pehchan_id: string | null;
  gi_tag_name: string | null;
  gi_certified: boolean;
  years_experience: number | null;
  skills_summary: string | null;
}

export interface QuickArtisanOnboardingForm {
  name: string | null;
  phone: string | null;
  pehchan_id: string | null;
}

export interface ExtractedFieldMeta {
  value: any;
  confidence: number;
  source_segment: string | null;
}

export interface ExtractedEntity {
  type: string;
  value: string;
  confidence: number;
}

export interface ExtractionResponse {
  form_data: ArtisanOnboardingForm;
  quick_form_data?: QuickArtisanOnboardingForm;
  field_metadata: Record<string, ExtractedFieldMeta>;
  model_used: string;
  processing_time_ms: number;
  overall_confidence: number;
  extracted_entities: ExtractedEntity[];
}

export interface SamplePrompt {
  id: string;
  title: string;
  dialect: string;
  category: string;
  text: string;
}
