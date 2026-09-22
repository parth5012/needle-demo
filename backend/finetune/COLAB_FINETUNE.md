# Fine-tune on Google Colab (GPU) — MAIN onboarding

Why Colab: local CPU training works but is slow (~1 h for 240 steps, batch-size 1,
and batch-size 16 OOMs). A free T4 GPU trains the same job in ~10–25 min and
allows bigger batches.

What you train: LoRA adapter on `backend/finetune/artisan_main.jsonl`
(88 examples: full narratives + inference-shaped chunks, MAIN 5-field schema),
then export merged `needle3-artisan.cact`.

## 1. Open a GPU notebook
1. Go to https://colab.research.google.com → New notebook.
2. Runtime → Change runtime type → **T4 GPU** (free tier is fine).

## 2. Run these cells in order

**Cell 1 — clone repo + install deps (~3–5 min):**
```python
!git clone https://github.com/parth5012/needle-demo /content/needle-demo
%cd /content/needle-demo/backend
!pip -q install cactus-needle "jax[cuda12]" optax flax sentencepiece safetensors
```

**Cell 2 — sanity check (GPU visible, data present):**
```python
import jax
print("backend:", jax.default_backend())  # want: gpu
!ls -la finetune/artisan_main.jsonl && wc -l finetune/artisan_main.jsonl
```

**Cell 3 — train (~10–25 min on T4):**
```python
!mkdir -p checkpoints
!needle finetune finetune/artisan_main.jsonl \
    --out checkpoints/needle_artisan_lora.safetensors \
    --epochs 3 --lora-rank 16 --val-split 0.1 --batch-size 4
```
- T4 16 GB usually fits `--batch-size 4`. If you hit OOM, drop to `--batch-size 2`
  (slower, same result). If it still OOMs, use `--batch-size 1`.
- Watch `val`: it should end **below where it started** (round reference:
  1.34 → 1.22 on mixed data). If val climbs while train loss falls, stop —
  that's overfitting; reduce `--epochs` to 2.
- First run downloads the base checkpoint (~230 MB) from Hugging Face automatically.

**Cell 4 — export merged weights (~1–2 min):**
```python
!needle build --lora checkpoints/needle_artisan_lora.safetensors \
    --out checkpoints/needle3-artisan.cact
!ls -la checkpoints/
```

**Cell 5 — quick inference check (optional, runs on Colab GPU/CPU):**
```python
from needle import Needle
import json
agent = Needle(weights="checkpoints/needle3-artisan.cact",
               tools=json.load(open("finetune/artisan_schema.json")),
               system=open("finetune/artisan_schema.json").read() and
               "Extract the artisan's personal identity details from the narrative. "
               "The name is the person's full personal name, never a greeting "
               "(never namaste, namaskara, pranam, hello, vanakkam) and never a place, "
               "cluster, or craft. Copy phone numbers, PIN codes, and government IDs "
               "exactly as written.")
text = ("Humar naam Gauri Devi ba. Contact number 9876543210 ba aur humar "
        "security pin 1234. Humaar government Pehchan ID PEH-IND-88320 ha aur "
        "TRIFED ID TRIFED-UP-VNS-1049.")
print(agent.complete(text, max_new_tokens=256)["function_calls"])
# expect: name Gauri Devi, phone 9876543210, pin 1234,
#         pehchan PEH-IND-88320, trifed TRIFED-UP-VNS-1049
```

**Cell 6 — download the model to your laptop:**
```python
from google.colab import files
files.download("checkpoints/needle3-artisan.cact")
# optional: files.download("checkpoints/needle_artisan_lora.safetensors")
```

## 3b. Score on the eval suite (do this before downloading)
`finetune/eval_cases.json` holds 35 ground-truth cases: 4 UI presets, 8 dialect
variants (Bhojpuri/Awadhi/Tamil/Gujarati/Telugu/Marwari/Nagpuri/Punjabi/Bengali/
Malwi/Maithili), phone formats (+91/spaces/dashes), PIN phrasings, reversed ID
order, same-sentence IDs, greetings, single-word/three-part names, partial
fields (missing → null), 3 off-topic refusals, year-count traps, PIN-before-phone,
GI-tail long narratives, pipe-delimited input.

**Cell 5b — run the scorer (needs `pydantic`: `!pip -q install pydantic`):**
```python
!python /content/needle-demo/backend/finetune/eval.py \
    --weights /content/needle-demo/backend/checkpoints/needle3-artisan.cact
```
Output: `cases 5/5 perfect: X/35`, per-field accuracy, and expected→got for
every miss. Local baseline (round-3 weights + hardened validators):
**23/35**, with phone/pin/trifed at 97% and misses clustering on `pehchan_id`
in long two-ID dialect sentences (71%) and name edge cases (91%) — those two
patterns are what the `LONG_TWO_ID` rows and the Nomoshkar/three-part-name/
pin-first/terse rows in `build_dataset.py` target, so gains here are your
green light. To debug one case: add `--limit 5` or narrow by editing the
JSON. To compare base vs tuned, run once without `--weights` (base) and once
with (tuned).

## 3. Install locally
1. Copy the downloaded `needle3-artisan.cact` into
   `D:\work\projects\needle-demo\backend\checkpoints\`
   (the engine auto-loads it; `$NEEDLE_MAIN_WEIGHTS` env var overrides).
2. Restart the backend **under the venv**:
   `D:\work\projects\needle-demo\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`
3. Check `/api/health` → `needle3.main_finetuned: true`.
4. Verify: `.\.venv\Scripts\python.exe -m pytest -q` (want 13 passed) and
   POST the Varanasi preset to `/api/extract-artisan` (want 5/5).

## 4. Iterate (the loop that got us here)
1. Failing case? Add it (and 2–3 siblings: reversed ID order, dialect
   rephrasing) to `build_dataset.py` (`BASE` for full narratives,
   `LONG_TWO_ID` for two-ID dialect sentences).
2. Regenerate: `python finetune/build_dataset.py`, commit + push the JSONL.
3. Re-run Colab cells 3–6 (pull first: `!git -C /content/needle-demo pull`).
4. Re-verify locally (step 3.4). Keep the weights that score best.

## Notes / pitfalls
- Tuned `.cact` files carry **no confidence head** → engine reports
  `confidence: None` and falls back to 0.95 clamped to 85–99%. Expected.
- Never commit the 230 MB base `checkpoints/needle3.safetensors` (GitHub
  caps files at 100 MB; it re-downloads automatically). Only
  `needle3-artisan.cact` (~60 MB) + optionally the LoRA adapter (~8 MB).
- Colab free GPUs have usage limits; if T4 is unavailable, CPU runtime also
  works (same commands, ~1 h, use `--batch-size 1`).
