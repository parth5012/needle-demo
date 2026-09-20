import time
import logging
from typing import Dict, Any, Tuple, Optional
from app.schemas import (
    ArtisanOnboardingForm,
    QuickArtisanOnboardingForm,
    ExtractionRequest,
    ExtractionResponse,
    ExtractedFieldMeta,
)

logger = logging.getLogger(__name__)

# Check if official needle 3 neural engine package is available
try:
    import needle
    NEEDLE3_AVAILABLE = True
except ImportError:
    NEEDLE3_AVAILABLE = False

class Needle3InferenceEngine:
    """
    Needle 3 Neural Inference & Extraction Engine.
    
    Uses Cactus Compute Needle 3 (`cactus-needle`) quantized foundation model weights (needle3.cact)
    for on-device structured JSON grammar extraction, with fallback contextual fusion.
    """

    KNOWN_CLUSTERS = {
        "varanasi": ("Uttar Pradesh", "Varanasi", "Varanasi Handloom Cluster", "Banarasi Brocade & Saree (GI Reg #39)", "Handloom Textile"),
        "channapatna": ("Karnataka", "Ramanagara", "Channapatna Craft Cluster", "Channapatna Toys and Dolls (GI Reg #16)", "Wooden Craft & Toys"),
        "madhubani": ("Bihar", "Madhubani", "Mithila Painting Cluster", "Madhubani Paintings (GI Reg #105)", "Folk Art & Painting"),
        "pochampally": ("Telangana", "Yadadri Bhuvanagiri", "Pochampally Ikat Cluster", "Pochampally Ikat (GI Reg #4)", "Handloom Textile"),
        "kutch": ("Gujarat", "Kutch", "Kutch Embroidery & Rogan Cluster", "Kutch Embroidery (GI Reg #152)", "Embroidery & Needlework"),
        "moradabad": ("Uttar Pradesh", "Moradabad", "Moradabad Brassware Cluster", "Moradabad Metal Craft (GI Reg #243)", "Metal Craft"),
        "bagru": ("Rajasthan", "Jaipur", "Bagru Hand Block Printing Cluster", "Bagru Hand Block Print (GI Reg #199)", "Block Print & Textile"),
        "kondapalli": ("Andhra Pradesh", "Krishna", "Kondapalli Bommallu Cluster", "Kondapalli Toys (GI Reg #19)", "Wooden Craft & Toys"),
    }

    def extract(self, req: ExtractionRequest) -> ExtractionResponse:
        start_time = time.perf_counter()
        text = req.text.strip()
        fields: Dict[str, ExtractedFieldMeta] = {}
        entities = []
        model_tag = "needle-3-neural (needle3.cact)"

        # 1. Neural Extraction using official Cactus Needle 3 model
        neural_quick_extracted = None
        if NEEDLE3_AVAILABLE:
            try:
                # Run Needle 3 structured extraction with schema
                extracted_obj = needle.extract(text, schema=QuickArtisanOnboardingForm, strict=False)
                if extracted_obj:
                    if hasattr(extracted_obj, 'name') and extracted_obj.name:
                        fields["name"] = ExtractedFieldMeta(value=extracted_obj.name, confidence=0.96, source_segment=f"Needle3: {extracted_obj.name}")
                        entities.append({"type": "PERSON_NAME", "value": extracted_obj.name, "confidence": 0.96})
                    if hasattr(extracted_obj, 'phone') and extracted_obj.phone:
                        fields["phone"] = ExtractedFieldMeta(value=extracted_obj.phone, confidence=0.98, source_segment=f"Needle3: {extracted_obj.phone}")
                        entities.append({"type": "PHONE_NUMBER", "value": extracted_obj.phone, "confidence": 0.98})
                    if hasattr(extracted_obj, 'pehchan_id') and extracted_obj.pehchan_id:
                        fields["pehchan_id"] = ExtractedFieldMeta(value=extracted_obj.pehchan_id, confidence=0.97, source_segment=f"Needle3: {extracted_obj.pehchan_id}")
                        entities.append({"type": "PEHCHAN_ID", "value": extracted_obj.pehchan_id, "confidence": 0.97})
                    neural_quick_extracted = extracted_obj
            except Exception as e:
                logger.warning(f"Needle 3 direct inference exception: {e}, using contextual fallback")

        # 2. Contextual & Domain Fallback Encoders for Heritage Specs
        if "name" not in fields:
            name_val, name_conf, name_seg = self._extract_name(text)
            if name_val:
                fields["name"] = ExtractedFieldMeta(value=name_val, confidence=name_conf, source_segment=name_seg)
                entities.append({"type": "PERSON_NAME", "value": name_val, "confidence": name_conf})

        if "phone" not in fields:
            phone_val, phone_conf, phone_seg = self._extract_phone(text)
            if phone_val:
                fields["phone"] = ExtractedFieldMeta(value=phone_val, confidence=phone_conf, source_segment=phone_seg)
                entities.append({"type": "PHONE_NUMBER", "value": phone_val, "confidence": phone_conf})

        if "pehchan_id" not in fields:
            pehchan_val, pehchan_conf, pehchan_seg = self._extract_pehchan(text)
            if pehchan_val:
                fields["pehchan_id"] = ExtractedFieldMeta(value=pehchan_val, confidence=pehchan_conf, source_segment=pehchan_seg)
                entities.append({"type": "PEHCHAN_ID", "value": pehchan_val, "confidence": pehchan_conf})

        pin_val, pin_conf, pin_seg = self._extract_pin(text)
        if pin_val:
            fields["pin"] = ExtractedFieldMeta(value=pin_val, confidence=pin_conf, source_segment=pin_seg)
            entities.append({"type": "SECURITY_PIN", "value": pin_val, "confidence": pin_conf})

        trifed_val, trifed_conf, trifed_seg = self._extract_trifed(text)
        if trifed_val:
            fields["trifed_id"] = ExtractedFieldMeta(value=trifed_val, confidence=trifed_conf, source_segment=trifed_seg)
            entities.append({"type": "TRIFED_ID", "value": trifed_val, "confidence": trifed_conf})

        # Location & Cluster Mapping
        loc_data = self._extract_cluster_and_location(text)
        for k, (v, c, seg) in loc_data.items():
            if v:
                fields[k] = ExtractedFieldMeta(value=v, confidence=c, source_segment=seg)
                entities.append({"type": k.upper(), "value": v, "confidence": c})

        dialect_val, dialect_conf, dialect_seg = self._extract_dialect(text)
        fields["primary_dialect"] = ExtractedFieldMeta(value=dialect_val, confidence=dialect_conf, source_segment=dialect_seg)

        craft_val, craft_conf, craft_seg = self._extract_craft(text, fields.get("craft_category"))
        if craft_val:
            fields["craft_category"] = ExtractedFieldMeta(value=craft_val, confidence=craft_conf, source_segment=craft_seg)
            entities.append({"type": "CRAFT_CATEGORY", "value": craft_val, "confidence": craft_conf})

        gi_name, gi_conf, gi_seg = self._extract_gi(text, fields.get("gi_tag_name"))
        is_gi = bool(gi_name)
        fields["gi_certified"] = ExtractedFieldMeta(value=is_gi, confidence=0.95 if is_gi else 0.7)
        if gi_name:
            fields["gi_tag_name"] = ExtractedFieldMeta(value=gi_name, confidence=gi_conf, source_segment=gi_seg)
            entities.append({"type": "GI_TAG", "value": gi_name, "confidence": gi_conf})

        exp_val, exp_conf, exp_seg = self._extract_experience(text)
        if exp_val is not None:
            fields["years_experience"] = ExtractedFieldMeta(value=exp_val, confidence=exp_conf, source_segment=exp_seg)

        fields["skills_summary"] = ExtractedFieldMeta(
            value=self._generate_summary(text, fields),
            confidence=0.88,
            source_segment=text[:120] + "..." if len(text) > 120 else text
        )

        def get_val(key: str, default: Any = None) -> Any:
            meta = fields.get(key)
            return meta.value if meta and meta.value is not None else default

        form = ArtisanOnboardingForm(
            name=get_val("name"),
            phone=get_val("phone"),
            pin=get_val("pin"),
            primary_dialect=get_val("primary_dialect", "hindi"),
            craft_category=get_val("craft_category"),
            state=get_val("state"),
            district=get_val("district"),
            cluster_name=get_val("cluster_name"),
            trifed_id=get_val("trifed_id"),
            pehchan_id=get_val("pehchan_id"),
            gi_tag_name=get_val("gi_tag_name"),
            gi_certified=get_val("gi_certified", False),
            years_experience=get_val("years_experience"),
            skills_summary=get_val("skills_summary"),
        )

        quick_form = QuickArtisanOnboardingForm(
            name=get_val("name"),
            phone=get_val("phone"),
            pehchan_id=get_val("pehchan_id"),
        )

        confidences = [f.confidence for f in fields.values() if f.value is not None]
        overall_conf = round(sum(confidences) / max(len(confidences), 1), 2)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ExtractionResponse(
            form_data=form,
            quick_form_data=quick_form,
            field_metadata=fields,
            model_used=model_tag if NEEDLE3_AVAILABLE else "needle-3-heuristic",
            processing_time_ms=elapsed_ms,
            overall_confidence=overall_conf,
            extracted_entities=entities,
        )

    STOP_NAME_WORDS = {"ba", "hai", "hu", "hain", "hove", "haye", "ani", "se", "aur", "and", "from", "in", "with", "a", "an", "the", "ki", "ka", "ke", "contact", "phone", "at", "to", "please", "call"}

    def _extract_name(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        import re
        patterns = [
            r"(?:my name is|i am|i'm|im|mera naam|naam|humar naam|name is)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})(?:,|\.|\s+contact|\s+phone|\s+from|\s+and|\s+with|\s+call|\s+pin|\s+pehchan|$)",
            r"(?:artisan|shilpkar|weaver|craftsperson)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                raw_name = m.group(1).strip()
                name_parts = [w for w in raw_name.split() if w.lower() not in self.STOP_NAME_WORDS]
                if name_parts:
                    cleaned_name = " ".join(name_parts).title()
                    return cleaned_name, 0.94, m.group(0)

        words = text.split()
        if len(words) >= 2 and words[0].istitle() and words[1].istitle() and not words[0].lower() in ["i", "my", "hello", "namaste", "we", "pranam"]:
            return f"{words[0]} {words[1]}", 0.75, f"{words[0]} {words[1]}"

        return None, 0.0, None

    def _extract_phone(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        import re
        match = re.search(r"(?:phone|mobile|contact|call|number|no\.?)?[:\s]*([6-9]\d{9})", text, re.IGNORECASE)
        if match:
            return match.group(1), 0.98, match.group(0)
        return None, 0.0, None

    def _extract_pin(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        import re
        match = re.search(r"(?:pin|passcode|security pin)[:\s]*(\d{4})\b", text, re.IGNORECASE)
        if match:
            return match.group(1), 0.95, match.group(0)
        return None, 0.0, None

    def _extract_pehchan(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        import re
        match = re.search(r"(?:pehchan|pehchan\s*card|pehchan\s*id)[:\s]*([A-Z0-9\-_]{6,20})", text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), 0.96, match.group(0)
        match_general = re.search(r"\b(PEH-[A-Z0-9\-]+)\b", text, re.IGNORECASE)
        if match_general:
            return match_general.group(1).upper(), 0.97, match_general.group(0)
        return None, 0.0, None

    def _extract_trifed(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        import re
        match = re.search(r"(?:trifed|trifed\s*id)[:\s]*([A-Z0-9\-_]{6,25})", text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), 0.96, match.group(0)
        match_general = re.search(r"\b(TRIFED-[A-Z0-9\-]+)\b", text, re.IGNORECASE)
        if match_general:
            return match_general.group(1).upper(), 0.97, match_general.group(0)
        return None, 0.0, None

    def _extract_cluster_and_location(self, text: str) -> Dict[str, Tuple[Optional[str], float, Optional[str]]]:
        import re
        results: Dict[str, Tuple[Optional[str], float, Optional[str]]] = {
            "state": (None, 0.0, None),
            "district": (None, 0.0, None),
            "cluster_name": (None, 0.0, None),
            "gi_tag_name": (None, 0.0, None),
            "craft_category": (None, 0.0, None),
        }
        lower = text.lower()
        for key, (state, district, cluster, gi_tag, craft) in self.KNOWN_CLUSTERS.items():
            if key in lower:
                results["state"] = (state, 0.96, key)
                results["district"] = (district, 0.96, key)
                results["cluster_name"] = (cluster, 0.96, key)
                results["gi_tag_name"] = (gi_tag, 0.95, key)
                results["craft_category"] = (craft, 0.92, key)
                return results

        state_match = re.search(r"(?:state|from|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?:state)?", text)
        if state_match:
            results["state"] = (state_match.group(1).strip(), 0.85, state_match.group(0))

        district_match = re.search(r"(?:district|city|village|town|near)\s+([A-Z][a-z]+)", text)
        if district_match:
            results["district"] = (district_match.group(1).strip(), 0.85, district_match.group(0))

        return results

    def _extract_dialect(self, text: str) -> Tuple[str, float, Optional[str]]:
        DIALECT_KEYWORDS = {
            "bhojpuri": "bhojpuri",
            "humar": "bhojpuri",
            "hamar": "bhojpuri",
            "bani": "bhojpuri",
            "maithili": "maithili",
            "kannada": "kannada",
            "telugu": "telugu",
            "gujarati": "gujarati",
            "marathi": "marathi",
            "hindi": "hindi",
            "rajasthani": "rajasthani",
            "marwari": "rajasthani",
            "tamil": "tamil",
            "bengali": "bengali",
            "odiya": "odia",
        }
        lower = text.lower()
        for kw, dialect in DIALECT_KEYWORDS.items():
            if kw in lower:
                return dialect, 0.92, kw
        return "hindi", 0.70, None

    def _extract_craft(self, text: str, existing_meta: Optional[ExtractedFieldMeta] = None) -> Tuple[Optional[str], float, Optional[str]]:
        if existing_meta and existing_meta.value:
            return existing_meta.value, existing_meta.confidence, existing_meta.source_segment

        CRAFT_KEYWORDS = {
            "saree": "Handloom Textile",
            "handloom": "Handloom Textile",
            "silk": "Handloom Textile",
            "brocade": "Handloom Textile",
            "weaving": "Handloom Textile",
            "toy": "Wooden Craft & Toys",
            "wooden": "Wooden Craft & Toys",
            "lacquer": "Wooden Craft & Toys",
            "painting": "Folk Art & Painting",
            "mithila": "Folk Art & Painting",
            "madhubani": "Folk Art & Painting",
            "brass": "Metal Craft",
            "metal": "Metal Craft",
            "bell metal": "Metal Craft",
            "embroidery": "Embroidery & Needlework",
            "rogan": "Folk Art & Painting",
            "pottery": "Clay & Terracotta Craft",
            "terracotta": "Clay & Terracotta Craft",
            "block print": "Block Print & Textile",
        }
        lower = text.lower()
        for kw, cat in CRAFT_KEYWORDS.items():
            if kw in lower:
                return cat, 0.90, kw
        return "Handicraft", 0.60, None

    def _extract_gi(self, text: str, existing_meta: Optional[ExtractedFieldMeta] = None) -> Tuple[Optional[str], float, Optional[str]]:
        import re
        if existing_meta and existing_meta.value:
            return existing_meta.value, existing_meta.confidence, existing_meta.source_segment

        gi_match = re.search(r"(?:gi tag|gi certified|gi registration|geographical indication)[:\s]*([^\n,\.]+)", text, re.IGNORECASE)
        if gi_match:
            return gi_match.group(1).strip(), 0.92, gi_match.group(0)

        if "gi" in text.lower() or "geographical indication" in text.lower():
            return "GI Registered Craft", 0.80, "gi reference"

        return None, 0.0, None

    def _extract_experience(self, text: str) -> Tuple[Optional[int], float, Optional[str]]:
        import re
        match = re.search(r"(\d+)\s*(?:\+|plus)?\s*(?:years|year|saal|baras|varshe)", text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1)), 0.95, match.group(0)
            except ValueError:
                pass
        return None, 0.0, None

    def _generate_summary(self, text: str, fields: Dict[str, ExtractedFieldMeta]) -> str:
        name_meta = fields.get("name")
        craft_meta = fields.get("craft_category")
        cluster_meta = fields.get("cluster_name")
        exp_meta = fields.get("years_experience")

        name = (name_meta.value if name_meta and name_meta.value else "Artisan")
        craft = (craft_meta.value if craft_meta and craft_meta.value else "Handicrafts")
        cluster = (cluster_meta.value if cluster_meta and cluster_meta.value else "")
        exp = (exp_meta.value if exp_meta and exp_meta.value else None)

        summary = f"{name} is a skilled artisan practicing {craft}"
        if cluster:
            summary += f" rooted in {cluster}"
        if exp:
            summary += f" with over {exp} years of master craftsmanship"
        summary += "."
        return summary
