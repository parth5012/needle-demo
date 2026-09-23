import json
import os
import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
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


def _load_main_tool_schema() -> Optional[Dict[str, Any]]:
    """Load the MAIN tool schema from finetune/artisan_schema.json.

    That file is the single source of truth shared with training
    (build_dataset.py embeds it in every finetune row). The finetuned model
    is extremely sensitive to prompt drift: an inline schema whose phone
    description gained an extra clause made the r4 model drop pehchan_id on
    every decode (0/4 vs 4/4 with the file schema). Loading the file keeps
    runtime prompts byte-identical to training prompts.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "finetune", "artisan_schema.json"))
    try:
        with open(path, encoding="utf-8") as f:
            schema = json.load(f)
        if isinstance(schema, dict):
            return schema
    except (OSError, ValueError) as e:
        logger.warning("[NEEDLE_3_STATUS] Could not load %s: %s", path, e)
    return None

_SHARED_QUICK_NEURAL_AGENT = None
_QUICK_NEURAL_AGENT_INITIALIZED = False

# System prompt steering the small model away from its most common mistake:
# copying a leading greeting ("Namaskara", "Pranam", ...) into the name slot.
_NEURAL_SYSTEM = (
    "Extract the artisan's personal identity details from the narrative. "
    "The name is the person's full personal name, never a greeting "
    "(never namaste, namaskara, pranam, hello, vanakkam) and never a place, "
    "cluster, or craft. Copy phone numbers, PIN codes, and government IDs "
    "exactly as written."
)

# Leading greeting prefix stripped from the text sent to the model so the
# greeting token never competes with the real name. This shapes the prompt
# only; no field values are extracted by it.
_LEADING_GREETING_RE = re.compile(
    r"^(?:namaste|namaskara|pranam|hello|hi|vanakkam|nomoshkar|johar|kem\s+cho|radhe|ram\s+ram"
    r"|sat\s+sri\s+akal|namaskar|good\s+(?:morning|afternoon|evening))[\s,!.]*",
    re.IGNORECASE,
)

_GREETING_WORDS = {
    "namaskara", "namaskar", "namaste", "pranam", "hello", "hi",
    "vanakkam", "nomoshkar", "johar", "kem cho", "radhe", "ram ram",
}
# Single-word tokens that can never be (part of) a person's name: places,
# crafts, and Hindi/Bhojpuri pronouns/possessives ("humar naam ..." = "my name ...").
_NON_NAME_TOKENS = {
    "cluster", "craft", "crafts", "handloom", "textile", "society", "ltd",
    "village", "district", "humar", "hamar", "humara", "hamara",
    "humaar", "hamaar", "humaara", "hamaara", "humra", "hamra",
    "hamre", "humre", "tumhar", "tumhara", "tohar", "tor", "mor",
    "mera", "meri", "mere", "naam", "main", "hum", "mai", "mein", "mujhe",
    "aapka", "aapke", "order", "orders", "applique",
}


def _strip_leading_greeting(text: str) -> str:
    """Remove a leading greeting so the model sees the person's name first."""
    stripped = _LEADING_GREETING_RE.sub("", text.strip()).strip()
    return stripped or text.strip()


def _chunk_text(text: str, max_chars: int = 220, overlap: int = 40) -> List[str]:
    """Split a narrative into overlapping chunks on sentence boundaries.

    Needle 3 has a small sliding KV window, so long narratives evict the very
    content to extract and the model rambles until its token budget is
    exhausted. Short chunks decode reliably; results are unioned by the caller.
    Chunks start at sentence boundaries so a chunk never opens mid-sentence
    (fragments make the model hallucinate name/phone from stray words and hide
    fields like the phone number in a boundary gap).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return _word_chunks(text, max_chars, overlap)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for sent in sentences:
        if len(sent) > max_chars:
            # Over-long single sentence: flush current, word-split the sentence.
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            sub = _word_chunks(sent, max_chars, overlap)
            chunks.extend(sub[:-1])
            cur, cur_len = ([sub[-1]], len(sub[-1])) if sub else ([], 0)
            continue
        add = len(sent) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append(" ".join(cur))
            # Overlap: carry the trailing sentence(s) that fit the budget.
            tail: List[str] = []
            tail_len = 0
            for ts in reversed(cur):
                if tail and tail_len + len(ts) + 1 > overlap:
                    break
                tail.append(ts)
                tail_len += len(ts) + 1
            cur = list(reversed(tail))
            cur_len = tail_len
            add = len(sent) + (1 if cur else 0)
        cur.append(sent)
        cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def _word_chunks(text: str, max_chars: int = 220, overlap: int = 40) -> List[str]:
    words = text.split()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append(" ".join(cur))
            tail: List[str] = []
            tail_len = 0
            for tw in reversed(cur):
                if tail_len + len(tw) + 1 > overlap:
                    break
                tail.append(tw)
                tail_len += len(tw) + 1
            cur = list(reversed(tail))
            cur_len = tail_len
            add = len(w) + (1 if cur else 0)
        cur.append(w)
        cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def _collect_neural_fields(
    calls: List[Any],
    *,
    allowed_keys: Tuple[str, ...],
    model_confidence: float,
    source_prefix: str,
    chunk_text: str = "",
) -> Tuple[Dict[str, ExtractedFieldMeta], List[Dict[str, Any]], List[str]]:
    """Validate raw tool calls into fields. Scans ALL returned calls
    (not just the first) and keeps the first valid value per key."""
    fields: Dict[str, ExtractedFieldMeta] = {}
    entities: List[Dict[str, Any]] = []
    extracted: List[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        args = call.get("arguments") or {}

        # Slot correction: if model placed an ID value in a non-ID slot (e.g. pin or phone)
        # or swapped slots, re-route it to its actual ID slot.
        clean_args = dict(args)
        for ak, av in list(clean_args.items()):
            slot = _id_slot(av)
            if slot and slot != ak:
                if slot not in clean_args or not clean_args[slot]:
                    clean_args[slot] = av
                if ak in ("pin", "phone"):
                    del clean_args[ak]

        for k in allowed_keys:
            if k in fields:
                continue
            val = clean_args.get(k)
            if val is None or str(val).strip() == "":
                continue
            str_val = str(val).strip()
            if k == "name":
                # A real name never contains a comma; the model appends
                # cluster/craft context after one ("Kiran Verma, Lucknow
                # Chikankari"). Keep the segment before the first comma.
                if "," in str_val:
                    str_val = str_val.split(",")[0].strip()
                    if not str_val:
                        continue
                lower_name = str_val.lower()
                name_tokens = set(re.findall(r"[a-z]+", lower_name))
                if lower_name in _GREETING_WORDS or (
                    name_tokens & (_GREETING_WORDS | _NON_NAME_TOKENS)
                ):
                    continue
            if k == "pin":
                # Normalize ("PIN-1234" -> "1234"); this is not the phone number.
                pin_digits = re.sub(r"\D", "", str_val)
                if len(pin_digits) != 4:
                    continue
                str_val = pin_digits
            if k == "phone":
                # Normalize ("+91 9876543210" -> "9876543210"); Indian mobiles
                # have 10 digits starting with 6-9. This rejects the model
                # confusing the 4-digit security PIN for the phone number.
                phone_digits = re.sub(r"\D", "", str_val)
                if len(phone_digits) > 10:
                    phone_digits = phone_digits[-10:]
                if len(phone_digits) != 10 or phone_digits[0] not in "6789":
                    continue
                str_val = phone_digits
            if k in ("pehchan_id", "trifed_id"):
                # IDs must look like identifiers, not stray words (e.g. "PIN").
                if len(str_val) < 6 or not re.search(r"[\d\-]", str_val):
                    continue
                # The model sometimes drops the alphabetic prefix ("GJ-KCH-1102"
                # instead of "TRIFED-GJ-KCH-1102"). Restore it: the value is
                # still 100% model-produced, only the known prefix is re-attached.
                # Never prepend when ANY known prefix is present — a cross-slot
                # value (PEH- under trifed_id) must keep its prefix so
                # _fix_swapped_ids can route it; prepending would fuse both
                # prefixes ("PEH-TRIFED-...") and destroy the signal.
                # Align to verbatim text ID if present in chunk_text
                if chunk_text:
                    prefix = "TRIFED-" if k == "trifed_id" else "PEH-"
                    cand_ids = re.findall(rf"\b{prefix}[A-Z0-9\-]+", chunk_text, re.IGNORECASE)
                    suffix = str_val.split("-")[-1]
                    for cand in cand_ids:
                        if cand.upper().endswith(suffix.upper()):
                            str_val = cand.upper()
                            break

                u = str_val.upper()
                if k == "trifed_id" and not u.startswith(("TRIFED-", "PEH-")):
                    str_val = "TRIFED-" + str_val
                elif k == "pehchan_id" and not u.startswith(("PEH-", "TRIFED-")):
                    str_val = "PEH-" + str_val
            conf = min(0.99, max(0.85, model_confidence))
            fields[k] = ExtractedFieldMeta(
                value=str_val,
                confidence=conf,
                source_segment=f"{source_prefix}: {k}={str_val}",
            )
            entities.append({"type": k.upper(), "value": str_val, "confidence": conf})
            extracted.append(k)
    return fields, entities, extracted


def _id_slot(value: Any) -> Optional[str]:
    """Which ID slot a model-produced value belongs to, by its own prefix."""
    u = str(value or "").upper()
    if u.startswith("PEH-"):
        return "pehchan_id"
    if u.startswith("TRIFED-"):
        return "trifed_id"
    return None


def _fix_swapped_ids(
    fields: Dict[str, ExtractedFieldMeta], entities: List[Dict[str, Any]]
) -> None:
    """The model sometimes files a PEH- ID under trifed_id and vice versa.
    Re-routes values by their own prefix. Values stay 100% model-produced;
    only the slot label is corrected."""
    for key in ("pehchan_id", "trifed_id"):
        meta = fields.get(key)
        if meta is None or meta.value is None:
            continue
        right = _id_slot(meta.value)
        if right is None or right == key:
            continue
        other = fields.get(right)
        if other is None or other.value is None:
            # Target slot empty: move the value over.
            fields[right] = ExtractedFieldMeta(
                value=meta.value,
                confidence=meta.confidence,
                source_segment=(meta.source_segment or "") + " [slot-corrected]",
            )
            del fields[key]
        elif _id_slot(other.value) != right:
            # Both misplaced: swap the values.
            meta.value, other.value = other.value, meta.value
            meta.source_segment = (meta.source_segment or "") + " [slot-corrected]"
            other.source_segment = (other.source_segment or "") + " [slot-corrected]"
        else:
            # Target slot already holds a correct value: drop the misplaced one.
            del fields[key]
    # Rebuild entities so they mirror the corrected fields.
    entities[:] = [
        {"type": k.upper(), "value": m.value, "confidence": m.confidence}
        for k, m in fields.items()
        if m.value is not None
    ]


def _neural_complete_best(
    agent: Any,
    text: str,
    *,
    allowed_keys: Tuple[str, ...],
    required_keys: Tuple[str, ...],
    source_prefix: str,
    max_new_tokens: int,
    max_attempts: int,
    log_tag: str,
    time_budget_s: float = 75.0,
) -> Tuple[Dict[str, ExtractedFieldMeta], List[Dict[str, Any]], List[str], float, float]:
    """Run the model over overlapping chunks of the text (short inputs decode
    reliably; long ones overflow the model's KV window and ramble), retrying
    each chunk and accumulating the union of valid fields across everything
    (first-valid-wins per key, longer-wins for IDs). Stops early once all
    `allowed_keys` are present or `time_budget_s` is exceeded."""
    budget_start = time.perf_counter()
    chunks = _chunk_text(text)
    merged_fields: Dict[str, ExtractedFieldMeta] = {}
    merged_entities: List[Dict[str, Any]] = []
    best_confidence = 0.0
    total_infer_ms = 0.0
    attempts = max_attempts
    for attempt in range(1, attempts + 1):
        if set(allowed_keys) <= set(merged_fields.keys()):
            break
        if (time.perf_counter() - budget_start) >= time_budget_s:
            break
        for ci, chunk in enumerate(chunks):
            if set(allowed_keys) <= set(merged_fields.keys()):
                break
            if attempt > 1:
                missing = set(allowed_keys) - set(merged_fields.keys())
                chunk_has_evidence = (
                    ("name" in missing)
                    or ("phone" in missing)
                    or ("pin" in missing and bool(re.search(r"\bpin\b", chunk, re.I)))
                    or ("pehchan_id" in missing and bool(re.search(r"peh|card", chunk, re.I)))
                    or ("trifed_id" in missing and bool(re.search(r"trifed", chunk, re.I)))
                )
                if not chunk_has_evidence:
                    continue
            elapsed_s = time.perf_counter() - budget_start
            if elapsed_s >= time_budget_s:
                logger.info(
                    "[%s] time budget %.0fs exceeded with %s; stopping.",
                    log_tag, time_budget_s, sorted(merged_fields.keys()),
                )
                break
            agent.reset()
            t_infer_start = time.perf_counter()
            res = agent.complete(chunk, max_new_tokens=max_new_tokens)
            infer_ms = round((time.perf_counter() - t_infer_start) * 1000, 2)
            total_infer_ms += infer_ms
            model_confidence = float(res.get("confidence") or 0.0)
            calls = res.get("function_calls") or res.get("suppressed_calls") or []
            # Re-collect against merged state so only genuinely new keys are added.
            # For IDs, a longer valid value replaces a shorter one (the model
            # sometimes emits truncated IDs like "PEH-IND-883").
            fresh_calls_fields, _, _ = _collect_neural_fields(
                calls,
                allowed_keys=tuple(
                    k for k in allowed_keys
                    if k not in merged_fields or k in ("pehchan_id", "trifed_id")
                ),
                model_confidence=model_confidence or 0.95,
                source_prefix=source_prefix,
                chunk_text=chunk,
            )
            for k, meta in fresh_calls_fields.items():
                if k not in merged_fields:
                    merged_fields[k] = meta
                    merged_entities.append(
                        {"type": k.upper(), "value": meta.value, "confidence": meta.confidence}
                    )
                elif k in ("pehchan_id", "trifed_id") and len(str(meta.value)) > len(
                    str(merged_fields[k].value)
                ):
                    merged_fields[k] = meta
                    for ent in merged_entities:
                        if ent["type"] == k.upper():
                            ent["value"] = meta.value
                            ent["confidence"] = meta.confidence
            best_confidence = max(best_confidence, model_confidence)
            logger.info(
                "[%s] chunk %d/%d attempt %d/%d in %.2f ms | new_valid=%s | merged=%s | confidence=%.3f",
                log_tag, ci + 1, len(chunks), attempt, attempts, infer_ms,
                sorted(fresh_calls_fields.keys()), sorted(merged_fields.keys()),
                model_confidence,
            )
            # Keep retrying while ANY allowed field is still missing (not just
            # the required core): with required_keys=("name", "phone") the loop
            # previously stopped after the first chunk that yielded those two,
            # leaving later ID-bearing chunks a single decode attempt — one bad
            # sample silently dropped pehchan_id. Bounded by max_attempts and
            # the time budget, so latency stays capped.
            if set(allowed_keys) <= set(merged_fields.keys()):
                break
        if (time.perf_counter() - budget_start) >= time_budget_s:
            break
    _fix_swapped_ids(merged_fields, merged_entities)
    return merged_fields, merged_entities, sorted(merged_fields.keys()), best_confidence, total_infer_ms

def _resolve_main_weights() -> Optional[str]:
    """Custom .cact for MAIN onboarding finetune (main only, quick untouched).

    Order: $NEEDLE_MAIN_WEIGHTS env var, then backend/checkpoints/needle3-artisan.cact.
    Returns None when absent so the caller falls back to base weights.
    """
    env_path = os.environ.get("NEEDLE_MAIN_WEIGHTS")
    if env_path and os.path.exists(env_path):
        return env_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "checkpoints", "needle3-artisan.cact"))
    alt = os.path.normpath(os.path.join(here, "..", "finetune", "needle3-artisan.cact"))
    for path in (candidate, alt):
        if os.path.exists(path):
            return path
    return None


def get_shared_quick_neural_agent():
    """
    Returns the singleton Cactus Needle 3 neural agent constrained strictly to 3 fields:
    name, phone, and pehchan_id.
    """
    global _SHARED_QUICK_NEURAL_AGENT, _QUICK_NEURAL_AGENT_INITIALIZED
    if _QUICK_NEURAL_AGENT_INITIALIZED:
        return _SHARED_QUICK_NEURAL_AGENT

    _QUICK_NEURAL_AGENT_INITIALIZED = True
    if NEEDLE3_AVAILABLE and needle is not None:
        try:
            logger.info("[NEEDLE_3_QUICK] Initializing dedicated 3-field Quick Needle 3 engine...")
            schema = {
                "name": "QuickArtisanOnboarding",
                "description": "Extract minimal artisan registration info: name, phone, and pehchan_id",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full personal name of the artisan person (never a greeting, place, or craft)"},
                        "phone": {"type": "string", "description": "10-digit mobile contact number starting with 6, 7, 8 or 9, copied exactly as written (this is not the 4-digit security PIN, and never a year-count like '18 years')"},
                        "pehchan_id": {"type": "string", "description": "Government Pehchan card identifier copied exactly as written (starts with PEH-)"},
                    }
                }
            }
            _SHARED_QUICK_NEURAL_AGENT = needle.Needle(tools=[schema], system=_NEURAL_SYSTEM)
            logger.info("[NEEDLE_3_QUICK] Dedicated 3-field Quick Needle 3 engine initialized successfully.")
        except Exception as e:
            logger.warning("[NEEDLE_3_QUICK] Failed to initialize 3-field Quick Needle 3 agent: %s", e)
            _SHARED_QUICK_NEURAL_AGENT = None
    return _SHARED_QUICK_NEURAL_AGENT

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
            # Schema MUST match finetune/artisan_schema.json byte-for-byte: it
            # is the prompt the finetune trained on (see _load_main_tool_schema).
            schema = _load_main_tool_schema()
            tuned_weights = _resolve_main_weights()
            _SHARED_NEURAL_AGENT = needle.Needle(
                tools=[schema],
                system=_NEURAL_SYSTEM,
                **({"weights": tuned_weights} if tuned_weights else {}),
            )
            if tuned_weights:
                logger.info("[NEEDLE_3_STATUS] Using finetuned MAIN weights: %s", tuned_weights)
            logger.info("[NEEDLE_3_STATUS] Cactus Needle 3 neural engine initialized successfully with ArtisanNeuralProfile grammar.")
        except Exception as e:
            logger.warning("[NEEDLE_3_STATUS] Failed to initialize Needle 3 neural agent: %s. Regex fallback will handle requests.", e)
            _SHARED_NEURAL_AGENT = None
    else:
        logger.info("[NEEDLE_3_STATUS] cactus-needle package not available. Using deterministic fallback engine.")
    return _SHARED_NEURAL_AGENT

class Needle3QuickInferenceEngine:
    """
    Dedicated 3-field Needle 3 Engine for Fast Onboarding.
    
    Extracts strictly:
    1. name
    2. phone
    3. pehchan_id
    
    Uses minimal token decode budget (max_new_tokens=128) to achieve low latency.
    """

    def __init__(self):
        self._fallback_extractor = DeterministicHeuristicExtractor()

    def extract(self, req: ExtractionRequest) -> ExtractionResponse:
        start_time = time.perf_counter()
        text = req.text.strip()
        fields: Dict[str, ExtractedFieldMeta] = {}
        entities = []
        neural_success = False
        neural_fields_extracted = []

        is_heuristic = getattr(req, "model_name", "needle-3-local") in (
            "needle-3-heuristic", "fast", "heuristic"
        )

        if NEEDLE3_AVAILABLE and not is_heuristic:
            try:
                agent = get_shared_quick_neural_agent()
                if agent is not None:
                    model_text = _strip_leading_greeting(text)
                    logger.info("[NEEDLE_3_QUICK] Running dedicated 3-field neural extraction (len=%d chars)", len(model_text))
                    (fields, entities, neural_fields_extracted,
                     model_confidence, infer_ms) = _neural_complete_best(
                        agent,
                        model_text,
                        allowed_keys=("name", "phone", "pehchan_id"),
                        required_keys=("name", "phone"),
                        source_prefix="Needle 3 Quick Neural",
                        max_new_tokens=128,
                        max_attempts=2,
                        log_tag="NEEDLE_3_QUICK",
                        time_budget_s=45.0,
                    )
                    if neural_fields_extracted:
                        neural_success = True
                        logger.info(
                            "[NEEDLE_3_QUICK] 3-field neural inference succeeded in %.2f ms | fields=%s | confidence=%.3f",
                            infer_ms, neural_fields_extracted, model_confidence
                        )
            except Exception as e:
                logger.warning("[NEEDLE_3_QUICK] Exception during 3-field neural extraction: %s", e)

        # Fallback strictly for the 3 fields
        # --- DISABLED (pure-neural mode): regex/heuristic fallback commented out
        # so inference uses only Needle 3 neural outputs. Missing fields stay None.
        # if "name" not in fields or fields["name"].value is None:
        #     n_val, n_conf, n_seg = self._fallback_extractor.extract_name(text)
        #     if n_val:
        #         fields["name"] = ExtractedFieldMeta(value=n_val, confidence=n_conf, source_segment=n_seg)
        #         entities.append({"type": "PERSON_NAME", "value": n_val, "confidence": n_conf})
        #
        # if "phone" not in fields or fields["phone"].value is None:
        #     p_val, p_conf, p_seg = self._fallback_extractor.extract_phone(text)
        #     if p_val:
        #         fields["phone"] = ExtractedFieldMeta(value=p_val, confidence=p_conf, source_segment=p_seg)
        #         entities.append({"type": "PHONE_NUMBER", "value": p_val, "confidence": p_conf})
        #
        # if "pehchan_id" not in fields or fields["pehchan_id"].value is None:
        #     peh_val, peh_conf, peh_seg = self._fallback_extractor.extract_pehchan(text)
        #     if peh_val:
        #         fields["pehchan_id"] = ExtractedFieldMeta(value=peh_val, confidence=peh_conf, source_segment=peh_seg)
        #         entities.append({"type": "PEHCHAN_ID", "value": peh_val, "confidence": peh_conf})

        def get_val(key: str, default: Any = None) -> Any:
            meta = fields.get(key)
            return meta.value if meta and meta.value is not None else default

        quick_form = QuickArtisanOnboardingForm(
            name=get_val("name"),
            phone=get_val("phone"),
            pehchan_id=get_val("pehchan_id"),
        )
        form = ArtisanOnboardingForm(
            name=get_val("name"),
            phone=get_val("phone"),
            pehchan_id=get_val("pehchan_id"),
            pin=None,
            primary_dialect="hindi",
            craft_category=None,
            state=None,
            district=None,
            cluster_name=None,
            trifed_id=None,
            gi_tag_name=None,
            gi_certified=False,
            years_experience=None,
            skills_summary=None,
        )

        confidences = [f.confidence for f in fields.values() if f.value is not None]
        overall_conf = round(sum(confidences) / max(len(confidences), 1), 2)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        model_tag = "needle-3-quick-neural (3-fields)" if neural_success else "needle-3-quick-heuristic"

        logger.info(
            "[NEEDLE_3_QUICK_SUMMARY] Completed in %.2f ms | model_used=%s | overall_confidence=%.2f | fields=%s",
            elapsed_ms, model_tag, overall_conf, list(fields.keys())
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
        self._quick_engine = Needle3QuickInferenceEngine()

    def extract(self, req: ExtractionRequest) -> ExtractionResponse:
        # Route to dedicated 3-field quick engine if requested
        if getattr(req, "quick_mode", False):
            return self._quick_engine.extract(req)

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
                    model_text = _strip_leading_greeting(text)
                    logger.info("[NEEDLE_3_NEURAL] Dispatching text to Needle 3 neural inference engine (len=%d chars)", len(model_text))
                    (fields, entities, neural_fields_extracted,
                     model_confidence, infer_ms) = _neural_complete_best(
                        agent,
                        model_text,
                        allowed_keys=("name", "phone", "pin", "pehchan_id", "trifed_id"),
                        required_keys=("name", "phone"),
                        source_prefix="Needle 3 Neural",
                        max_new_tokens=256,
                        max_attempts=3,
                        log_tag="NEEDLE_3_NEURAL",
                        time_budget_s=90.0,
                    )
                    if neural_fields_extracted:
                        neural_success = True
                        logger.info(
                            "[NEEDLE_3_NEURAL] Neural inference succeeded in %.2f ms | fields=%s | confidence=%.3f",
                            infer_ms,
                            neural_fields_extracted,
                            model_confidence,
                        )
                    else:
                        logger.info("[NEEDLE_3_NEURAL] Neural engine completed in %.2f ms with no valid identity fields.", infer_ms)
                else:
                    logger.info("[NEEDLE_3_NEURAL] Neural agent unavailable; pure-neural mode returns empty fields.")
            except Exception as e:
                logger.warning("[NEEDLE_3_NEURAL] Neural inference encountered exception: %s.", e)
        else:
            logger.info("[NEEDLE_3_NEURAL] Heuristic mode requested or neural engine not available; pure-neural mode returns empty fields.")

        # 2. Deterministic & Heritage Knowledge Fallback for Missing Fields
        # --- DISABLED (pure-neural mode): regex/heuristic fallback commented out
        # so inference uses only Needle 3 neural outputs. Missing fields stay None.
        # before_count = len(fields)
        # fields, entities = self._fallback_extractor.extract_missing_fields(text, fields, entities)
        # added_by_fallback = [k for k in fields.keys() if k not in neural_fields_extracted]
        #
        # if added_by_fallback:
        #     if neural_success:
        #         logger.info(
        #             "[NEEDLE_3_REGEX_FALLBACK] Enriched %d regional/domain fields via deterministic knowledge: %s",
        #             len(added_by_fallback),
        #             added_by_fallback
        #         )
        #     else:
        #         logger.info(
        #             "[NEEDLE_3_REGEX_FALLBACK] Pure deterministic regex extraction completed (%d fields): %s",
        #             len(fields),
        #             list(fields.keys())
        #         )

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
