# Fine-tune Needle 3 — MAIN onboarding (5 fields)

Scope: **main onboarding only** (`/api/extract-artisan` → `ArtisanNeuralProfile`:
`name, phone, pin, pehchan_id, trifed_id`). Quick 3-field flow untouched.

## Data
- `artisan_schema.json` — tool schema matching `get_shared_neural_agent()` in `backend/app/needle_engine.py`.
- `build_dataset.py` — deterministic generator (no API key needed). Re-run:
  `.\.venv\Scripts\python.exe finetune\build_dataset.py`
- `artisan_main.jsonl` — 39 examples: 4 repo fixtures + dialect variants
  (Bhojpuri/Maithili/Kannada/Gujarati/Tamil), phone-vs-PIN, PEH-/TRIFED- slot
  correctness, greeting guards, partial fields, off-topic refusals.
- Optional expansion (needs `OPENROUTER_API_KEY`):
  `.\.venv\Scripts\needle.exe generate-data --tools finetune\artisan_schema.json --num-samples 60`
  then append/merge into `artisan_main.jsonl`.

## Train (LoRA)
From `backend/`:
```
.\.venv\Scripts\needle.exe finetune finetune\artisan_main.jsonl --out checkpoints\needle_artisan_lora.safetensors --epochs 3 --lora-rank 16 --val-split 0.1
```
Tune `--epochs 2-5`, `--lora-rank 16`, `--lr`, `--batch-size` as needed. Base
checkpoint auto-downloads from HuggingFace on first run.

## Export merged weights
```
.\.venv\Scripts\needle.exe build --lora checkpoints\needle_artisan_lora.safetensors --out checkpoints\needle3-artisan.cact
```
Keep local + git-ignored (`*.cact`), or publish with `--upload` + `$NEEDLE_HF_REPO`.

## Wire-in
`get_shared_neural_agent()` checks in order:
1. `$NEEDLE_MAIN_WEIGHTS` env var (absolute .cact path),
2. `backend/checkpoints/needle3-artisan.cact`,
3. base weights (default `needle.Needle(tools, system)`).

No code change needed after training — just drop the file or set the env var
and restart the backend under the venv. Tuned weights without a confidence head
report `confidence: None`; the engine falls back to 0.95 and clamps display to
85–99% (see `_collect_neural_fields`).

## Evaluate
```
.\.venv\Scripts\python.exe -m pytest -q
npm run build   # from frontend/
```
Then live: `POST /api/extract-artisan` with the Varanasi Bhojpuri narrative —
expect 5/5 (Gauri Devi / 9876543210 / 1234 / PEH-IND-88320 / TRIFED-UP-VNS-1049).
