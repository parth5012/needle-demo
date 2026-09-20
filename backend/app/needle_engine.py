import time
import logging
from typing import Dict, Any, Optional
from app.schemas import (
    ArtisanOnboardingForm,
    QuickArtisanOnboardingForm,
    ExtractionRequest,
    ExtractionResponse,
    ExtractedFieldMeta,
)
from app.heuristic_engine import DeterministicHeuristicExtractor

logger = logging.getLogger("needle.engine")

# Check if official needle 3 neural engine package is available
try:
    import needle  # type: ignore
    NEEDLE3_AVAILABLE = True
except ImportError:
    needle = None  # type: ignore
    NEEDLE3_AVAILABLE = False

_SHARED_NEURAL_AGENT = None
_NEURAL_AGENT_INITIALIZED = False

def get_shared_neural_agent():
    """
    Returns the singleton Cactus Needle 3 neural agent, pre-compiling grammar once.
    """
    global _SHARED_NEURAL_AGENT, _NEURAL_AGENT_INITIALIZED
    if _NEURAL_AGENT_INITIALIZED:
        return _SHARED_NEURAL_AGENT

    _NEURAL_AGENT_INITIALIZED = True
    if NEEDLE3_AVAILABLE and needle is not None:
        try:
            logger.info("[NEEDLE_3_STATUS] Initializing Cactus Needle 3 neural engine (needle3.cact)...")
            schema = {
                "name": "ArtisanNeuralProfile",
                "description": "Extract artisan identity and registration details",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full name of artisan"},
                        "phone": {"type": "string", "description": "10-digit mobile phone number"},
                        "pin": {"type": "string", "description": "4-digit security PIN"},
                        "pehchan_id": {"type": "string", "description": "Government Pehchan card ID"},
                        "trifed_id": {"type": "string", "description": "Government TRIFED registration ID"},
                    }
                }
            }
            _SHARED_NEURAL_AGENT = needle.Needle(tools=[schema])
            logger.info("[NEEDLE_3_STATUS] Cactus Needle 3 neural engine initialized successfully with ArtisanNeuralProfile grammar.")
        except Exception as e:
            logger.warning("[NEEDLE_3_STATUS] Failed to initialize Needle 3 neural agent: %s. Regex fallback will handle requests.", e)
            _SHARED_NEURAL_AGENT = None
    else:
        logger.info("[NEEDLE_3_STATUS] cactus-needle package not available. Using deterministic fallback engine.")
    return _SHARED_NEURAL_AGENT

class Needle3InferenceEngine:
    """
    Needle 3 Neural Inference & Extraction Engine.
    
    Uses Cactus Compute Needle 3 (`cactus-needle`) quantized foundation model weights (needle3.cact)
    for on-device structured JSON grammar extraction. Offloads missing fields and offline modes to
    DeterministicHeuristicExtractor.
    """

    KNOWN_CLUSTERS = DeterministicHeuristicExtractor.KNOWN_CLUSTERS

    def __init__(self):
        self._fallback_extractor = DeterministicHeuristicExtractor()

    def extract(self, req: ExtractionRequest) -> ExtractionResponse:
        start_time = time.perf_counter()
        text = req.text.strip()
        fields: Dict[str, ExtractedFieldMeta] = {}
        entities = []
        neural_success = False
        neural_fields_extracted = []

        is_heuristic_requested = getattr(req, "model_name", "needle-3-local") in (
            "needle-3-heuristic", "fast", "heuristic"
        )

        # 1. Neural Extraction using official Cactus Needle 3 model weights
        if NEEDLE3_AVAILABLE and not is_heuristic_requested:
            try:
                agent = get_shared_neural_agent()
                if agent is not None:
                    agent.reset()
                    logger.info("[NEEDLE_3_NEURAL] Dispatching text to Needle 3 neural inference engine (len=%d chars)", len(text))
                    t_infer_start = time.perf_counter()
                    res = agent.complete(text, max_new_tokens=160)
                    infer_ms = round((time.perf_counter() - t_infer_start) * 1000, 2)
                    
                    model_confidence = float(res.get("confidence") or 0.95)
                    calls = res.get("function_calls") or res.get("suppressed_calls") or []
                    
                    if calls and isinstance(calls[0], dict):
                        args = calls[0].get("arguments") or {}
                        for k, val in args.items():
                            if val is not None and str(val).strip() != "":
                                str_val = str(val).strip()
                                # Validation guards to prevent hallucination spillover
                                GREETING_WORDS = {"namaskara", "namaste", "pranam", "hello", "hi", "vanakkam", "kem cho", "radhe"}
                                if k == "name":
                                    lower_name = str_val.lower()
                                    if lower_name in GREETING_WORDS or any(w in lower_name for w in ["cluster", "craft", "handloom", "textile", "society", "ltd"]):
                                        continue
                                if k == "pin" and (not str_val.isdigit() or len(str_val) != 4):
                                    continue
                                if k == "phone" and len([c for c in str_val if c.isdigit()]) != 10:
                                    continue

                                conf = min(0.99, max(0.85, model_confidence))
                                fields[k] = ExtractedFieldMeta(
                                    value=str_val,
                                    confidence=conf,
                                    source_segment=f"Needle 3 Neural: {k}={str_val}"
                                )
                                entities.append({
                                    "type": k.upper(),
                                    "value": str_val,
                                    "confidence": conf
                                })
                                neural_fields_extracted.append(k)

                        if neural_fields_extracted:
                            neural_success = True
                            logger.info(
                                "[NEEDLE_3_NEURAL] Neural inference succeeded in %.2f ms | fields=%s | confidence=%.3f | tps=%.1f",
                                infer_ms,
                                neural_fields_extracted,
                                model_confidence,
                                res.get("decode_tps", 0.0)
                            )
                        else:
                            logger.info("[NEEDLE_3_NEURAL] Neural engine completed in %.2f ms with no valid identity fields.", infer_ms)
                    else:
                        logger.info("[NEEDLE_3_NEURAL] Off-topic or empty call returned by model in %.2f ms.", infer_ms)
                else:
                    logger.info("[NEEDLE_3_REGEX_FALLBACK] Neural agent unavailable. Delegating directly to deterministic regex extractor.")
            except Exception as e:
                logger.warning("[NEEDLE_3_NEURAL] Neural inference encountered exception: %s. Falling back to deterministic extractor.", e)
        else:
            logger.info("[NEEDLE_3_REGEX_FALLBACK] Heuristic mode requested or neural engine not available.")

        # 2. Deterministic & Heritage Knowledge Fallback for Missing Fields
        before_count = len(fields)
        fields, entities = self._fallback_extractor.extract_missing_fields(text, fields, entities)
        added_by_fallback = [k for k in fields.keys() if k not in neural_fields_extracted]

        if added_by_fallback:
            if neural_success:
                logger.info(
                    "[NEEDLE_3_REGEX_FALLBACK] Enriched %d regional/domain fields via deterministic knowledge: %s",
                    len(added_by_fallback),
                    added_by_fallback
                )
            else:
                logger.info(
                    "[NEEDLE_3_REGEX_FALLBACK] Pure deterministic regex extraction completed (%d fields): %s",
                    len(fields),
                    list(fields.keys())
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
        model_tag = "needle-3-neural (needle3.cact)" if neural_success else "needle-3-heuristic"

        logger.info(
            "[NEEDLE_3_SUMMARY] Completed in %.2f ms | model_used=%s | overall_confidence=%.2f | total_fields=%d",
            elapsed_ms,
            model_tag,
            overall_conf,
            len(confidences)
        )

        return ExtractionResponse(
            form_data=form,
            quick_form_data=quick_form,
            field_metadata=fields,
            model_used=model_tag,
            processing_time_ms=elapsed_ms,
            overall_confidence=overall_conf,
            extracted_entities=entities,
        )
