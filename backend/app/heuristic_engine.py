import re
import logging
from typing import Dict, Any, Tuple, Optional, List
from app.schemas import ExtractedFieldMeta

logger = logging.getLogger(__name__)

class DeterministicHeuristicExtractor:
    """
    Deterministic rule-based and regex fallback extractor for Indian artisanal heritage data.
    
    Used strictly as fallback when Cactus Needle 3 neural engine is unavailable,
    or to fill regional cluster metadata not captured by the neural grammar.
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

    STOP_NAME_WORDS = {
        "ba", "hai", "hu", "hain", "hove", "haye", "ani", "se", "aur", "and",
        "from", "in", "with", "a", "an", "the", "ki", "ka", "ke", "contact",
        "phone", "at", "to", "please", "call"
    }

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

    def extract_missing_fields(
        self,
        text: str,
        fields: Dict[str, ExtractedFieldMeta],
        entities: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, ExtractedFieldMeta], List[Dict[str, Any]]]:
        """
        Fills any field gaps in the extracted record using regex and domain lookups.
        """
        if "name" not in fields or fields["name"].value is None:
            name_val, name_conf, name_seg = self.extract_name(text)
            if name_val:
                fields["name"] = ExtractedFieldMeta(value=name_val, confidence=name_conf, source_segment=name_seg)
                entities.append({"type": "PERSON_NAME", "value": name_val, "confidence": name_conf})

        if "phone" not in fields or fields["phone"].value is None:
            phone_val, phone_conf, phone_seg = self.extract_phone(text)
            if phone_val:
                fields["phone"] = ExtractedFieldMeta(value=phone_val, confidence=phone_conf, source_segment=phone_seg)
                entities.append({"type": "PHONE_NUMBER", "value": phone_val, "confidence": phone_conf})

        if "pehchan_id" not in fields or fields["pehchan_id"].value is None:
            pehchan_val, pehchan_conf, pehchan_seg = self.extract_pehchan(text)
            if pehchan_val:
                fields["pehchan_id"] = ExtractedFieldMeta(value=pehchan_val, confidence=pehchan_conf, source_segment=pehchan_seg)
                entities.append({"type": "PEHCHAN_ID", "value": pehchan_val, "confidence": pehchan_conf})

        if "pin" not in fields or fields["pin"].value is None:
            pin_val, pin_conf, pin_seg = self.extract_pin(text)
            if pin_val:
                fields["pin"] = ExtractedFieldMeta(value=pin_val, confidence=pin_conf, source_segment=pin_seg)
                entities.append({"type": "SECURITY_PIN", "value": pin_val, "confidence": pin_conf})

        if "trifed_id" not in fields or fields["trifed_id"].value is None:
            trifed_val, trifed_conf, trifed_seg = self.extract_trifed(text)
            if trifed_val:
                fields["trifed_id"] = ExtractedFieldMeta(value=trifed_val, confidence=trifed_conf, source_segment=trifed_seg)
                entities.append({"type": "TRIFED_ID", "value": trifed_val, "confidence": trifed_conf})

        # Location & Cluster Mapping
        if not any(k in fields and fields[k].value is not None for k in ("state", "district", "cluster_name")):
            loc_data = self.extract_cluster_and_location(text)
            for k, (v, c, seg) in loc_data.items():
                if v and (k not in fields or fields[k].value is None):
                    fields[k] = ExtractedFieldMeta(value=v, confidence=c, source_segment=seg)
                    entities.append({"type": k.upper(), "value": v, "confidence": c})

        if "primary_dialect" not in fields or fields["primary_dialect"].value is None:
            dialect_val, dialect_conf, dialect_seg = self.extract_dialect(text)
            fields["primary_dialect"] = ExtractedFieldMeta(value=dialect_val, confidence=dialect_conf, source_segment=dialect_seg)

        if "craft_category" not in fields or fields["craft_category"].value is None:
            craft_val, craft_conf, craft_seg = self.extract_craft(text, fields.get("craft_category"))
            if craft_val:
                fields["craft_category"] = ExtractedFieldMeta(value=craft_val, confidence=craft_conf, source_segment=craft_seg)
                entities.append({"type": "CRAFT_CATEGORY", "value": craft_val, "confidence": craft_conf})

        if "gi_certified" not in fields or fields["gi_certified"].value is None or "gi_tag_name" not in fields:
            gi_name, gi_conf, gi_seg = self.extract_gi(text, fields.get("gi_tag_name"))
            is_gi = bool(gi_name)
            if "gi_certified" not in fields or fields["gi_certified"].value is None:
                fields["gi_certified"] = ExtractedFieldMeta(value=is_gi, confidence=0.95 if is_gi else 0.7)
            if gi_name and ("gi_tag_name" not in fields or fields["gi_tag_name"].value is None):
                fields["gi_tag_name"] = ExtractedFieldMeta(value=gi_name, confidence=gi_conf, source_segment=gi_seg)
                entities.append({"type": "GI_TAG", "value": gi_name, "confidence": gi_conf})

        if "years_experience" not in fields or fields["years_experience"].value is None:
            exp_val, exp_conf, exp_seg = self.extract_experience(text)
            if exp_val is not None:
                fields["years_experience"] = ExtractedFieldMeta(value=exp_val, confidence=exp_conf, source_segment=exp_seg)

        if "skills_summary" not in fields or fields["skills_summary"].value is None:
            fields["skills_summary"] = ExtractedFieldMeta(
                value=self.generate_summary(text, fields),
                confidence=0.88,
                source_segment=text[:120] + "..." if len(text) > 120 else text
            )

        return fields, entities

    def extract_name(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
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

    def extract_phone(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        match = re.search(r"(?:phone|mobile|contact|call|number|no\.?)?[:\s]*([6-9]\d{9})", text, re.IGNORECASE)
        if match:
            return match.group(1), 0.98, match.group(0)
        return None, 0.0, None

    def extract_pin(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        match = re.search(r"(?:pin|passcode|security pin)[:\s]*(\d{4})\b", text, re.IGNORECASE)
        if match:
            return match.group(1), 0.95, match.group(0)
        return None, 0.0, None

    def extract_pehchan(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        # Prefer explicit PEH-... identifiers over loose "Pehchan <word>" captures.
        match_general = re.search(r"\b(PEH-[A-Z0-9\-]+)\b", text, re.IGNORECASE)
        if match_general:
            return match_general.group(1).upper(), 0.97, match_general.group(0)
        match = re.search(r"(?:pehchan|pehchan\s*card|pehchan\s*id)[:\s]*([A-Z0-9\-_]{6,20})", text, re.IGNORECASE)
        if match:
            token = match.group(1)
            # Guard: bare words (e.g. "card", "number") are not IDs; require a digit or hyphen.
            if re.search(r"[\d\-]", token):
                return token.upper(), 0.96, match.group(0)
        return None, 0.0, None

    def extract_trifed(self, text: str) -> Tuple[Optional[str], float, Optional[str]]:
        # Prefer explicit TRIFED-... identifiers over loose "TRIFED <word>" captures
        # (e.g. "TRIFED registration number ..." must not yield "REGISTRATION").
        match_general = re.search(r"\b(TRIFED-[A-Z0-9\-]+)\b", text, re.IGNORECASE)
        if match_general:
            return match_general.group(1).upper(), 0.97, match_general.group(0)
        match = re.search(r"(?:trifed|trifed\s*id)[:\s]*([A-Z0-9\-_]{6,25})", text, re.IGNORECASE)
        if match:
            token = match.group(1)
            # Guard: bare words (e.g. "registration") are not IDs; require a digit or hyphen.
            if re.search(r"[\d\-]", token):
                return token.upper(), 0.96, match.group(0)
        return None, 0.0, None

    def extract_cluster_and_location(self, text: str) -> Dict[str, Tuple[Optional[str], float, Optional[str]]]:
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

    def extract_dialect(self, text: str) -> Tuple[str, float, Optional[str]]:
        lower = text.lower()
        for kw, dialect in self.DIALECT_KEYWORDS.items():
            if kw in lower:
                return dialect, 0.92, kw
        return "hindi", 0.70, None

    def extract_craft(self, text: str, existing_meta: Optional[ExtractedFieldMeta] = None) -> Tuple[Optional[str], float, Optional[str]]:
        if existing_meta and existing_meta.value:
            return existing_meta.value, existing_meta.confidence, existing_meta.source_segment

        lower = text.lower()
        for kw, cat in self.CRAFT_KEYWORDS.items():
            if kw in lower:
                return cat, 0.90, kw
        return "Handicraft", 0.60, None

    def extract_gi(self, text: str, existing_meta: Optional[ExtractedFieldMeta] = None) -> Tuple[Optional[str], float, Optional[str]]:
        if existing_meta and existing_meta.value:
            return existing_meta.value, existing_meta.confidence, existing_meta.source_segment

        gi_match = re.search(r"(?:gi tag|gi certified|gi registration|geographical indication)[:\s]*([^\n,\.]+)", text, re.IGNORECASE)
        if gi_match:
            return gi_match.group(1).strip(), 0.92, gi_match.group(0)

        if "gi" in text.lower() or "geographical indication" in text.lower():
            return "GI Registered Craft", 0.80, "gi reference"

        return None, 0.0, None

    def extract_experience(self, text: str) -> Tuple[Optional[int], float, Optional[str]]:
        match = re.search(r"(\d+)\s*(?:\+|plus)?\s*(?:years|year|saal|baras|varshe)", text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1)), 0.95, match.group(0)
            except ValueError:
                pass
        return None, 0.0, None

    def generate_summary(self, text: str, fields: Dict[str, ExtractedFieldMeta]) -> str:
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
